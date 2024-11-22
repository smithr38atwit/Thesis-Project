import json
import os
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np
import torch as th
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from scipy import stats as sts


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [
            Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.train_collisions = []
        self.test_collisions = []
        self.train_deliveries = []
        self.test_deliveries = []

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_deliveries = [0 for _ in range(self.batch_size)]
        episode_collissions = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode
            )
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {"reward": [], "terminated": []}
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        episode_deliveries[idx] = data["info"].pop("deliveries", 0)
                        episode_collissions[idx] = data["info"].pop("collisions", 0)
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_deliveries = self.test_deliveries if test_mode else self.train_deliveries
        cur_collisions = self.test_collisions if test_mode else self.train_collisions
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        cur_deliveries.extend(episode_deliveries)
        cur_collisions.extend(episode_collissions)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            if self.args.evaluate:
                self._save_stats(cur_returns, cur_deliveries, cur_collisions)
            self._log(cur_returns, cur_stats, log_prefix, cur_deliveries, cur_collisions)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _save_stats(self, returns, deliveries, collisions):
        stats = {
            "metadata": {
                "model": self.args.friendly_name,
                "model_path": self.args.checkpoint_path,
                "env": self.args.env_args["key"],
                "time_limit": self.args.env_args["time_limit"],
                "num_episodes": self.args.test_nepisode,
            },
            "averages": {
                "rewards": {
                    "min": min(returns),
                    "max": max(returns),
                    "average": np.mean(returns),
                    "std": np.std(returns),
                    "confidence_interval": self._ci(returns),
                },
                "collisions": {
                    "min": min(collisions),
                    "max": max(collisions),
                    "average": np.mean(collisions),
                    "std": np.std(collisions),
                    "confidence_interval": (self._ci(collisions) if np.mean(collisions) > 0 else 0),
                },
                "deliveries": {
                    "min": min(deliveries),
                    "max": max(deliveries),
                    "average": np.mean(deliveries),
                    "std": np.std(deliveries),
                    "confidence_interval": self._ci(deliveries),
                },
            },
            "returns": returns,
            "collisions": collisions,
            "deliveries": deliveries,
        }

        # Log stats to file with incrementing suffix
        base_filename = f"/home/smithr38/Thesis-Project/stats/mappo/{self.args.friendly_name}.json"
        suffix = 0
        while os.path.exists(base_filename):
            suffix += 1
            base_filename = f"/home/smithr38/Thesis-Project/stats/mappo/{self.args.friendly_name}_{suffix}.json"

        with open(base_filename, "w") as f:
            json.dump(stats, f, indent=4)

    def _log(self, returns, stats, prefix, deliveries=None, collisions=None):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        self.logger.log_stat(prefix + "return_ci", self._ci(returns), self.t_env)
        self.logger.log_stat(prefix + "return_min", min(returns), self.t_env)
        self.logger.log_stat(prefix + "return_max", max(returns), self.t_env)
        returns.clear()
        if deliveries and self.args.evaluate:
            self.logger.log_stat(prefix + "delivery_mean", np.mean(deliveries), self.t_env)
            self.logger.log_stat(prefix + "delivery_std", np.std(deliveries), self.t_env)
            self.logger.log_stat(prefix + "delivery_ci", self._ci(deliveries), self.t_env)
            self.logger.log_stat(prefix + "delivery_min", min(deliveries), self.t_env)
            self.logger.log_stat(prefix + "delivery_max", max(deliveries), self.t_env)
            deliveries.clear()
        if collisions and self.args.evaluate:
            self.logger.log_stat(prefix + "collisions_mean", np.mean(collisions), self.t_env)
            self.logger.log_stat(prefix + "collision_std", np.std(collisions), self.t_env)
            self.logger.log_stat(
                prefix + "collision_ci", self._ci(collisions) if np.mean(collisions) > 0 else 0, self.t_env
            )
            self.logger.log_stat(prefix + "collision_min", min(collisions), self.t_env)
            self.logger.log_stat(prefix + "collision_max", max(collisions), self.t_env)
            collisions.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

    def _ci(self, data):
        mean = np.mean(data)
        if mean == 0:
            return 0
        sem = sts.sem(data)
        confidence_interval = sts.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
        return (confidence_interval[1] - confidence_interval[0]) / 2


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send({"state": env.get_state(), "avail_actions": env.get_avail_actions(), "obs": env.get_obs()})
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
