import gym
import numpy as np
import rware
import torch
from a2c import A2C
from wrappers import Monitor, RecordEpisodeStatistics, TimeLimit

from utils import save_animation

model_name = "seac/tiny_2ag_1p_20m"
path = f"/home/lambda10/rsmith_thesis/Thesis-Project/models/{model_name}"
env_name = "rware-tiny-2ag-v1"
time_limit = 500  # 25 for LBF

EPISODES = 1000
RECORD = False
RECORD_OUTLIERS = True
STATS = False

env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

all_rewards = []
all_collisions = []
all_deliveries = []
max_r, max_c, max_d = (-np.inf,) * 3
min_r, min_c, min_d = (np.inf,) * 3
for ep in range(EPISODES):
    print("Episode: " + str(ep + 1), end="\r")

    env = gym.make(env_name)
    if RECORD:
        env = Monitor(env, f"videos/test/{model_name}_wp/video_ep{ep+1}", mode="evaluation")
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)

    obs = env.reset()
    done = [False] * len(agents)

    frames = []
    while not all(done):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _, _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        if RECORD:
            env.render()
        elif RECORD_OUTLIERS:
            frames.append(env.render(mode="rgb_array"))
        obs, _, done, info = env.step(actions)
    obs = env.reset()
    env.close()

    episode_reward = info["episode_reward"]
    collisions = info["collisions"]
    deliveries = info["deliveries"]
    total_reward = sum(episode_reward)
    total_collisions = sum(collisions)
    total_deliveries = sum(deliveries)
    all_rewards.append(total_reward)
    all_collisions.append(total_collisions)
    all_deliveries.append(total_deliveries)
    # maxs and mins
    if max(episode_reward) > max_r:
        max_r = max(episode_reward)
    elif min(episode_reward) < min_r:
        min_r = min(episode_reward)
    if max(collisions) > max_c:
        max_c = max(collisions)
    elif min(collisions) < min_c:
        min_c = min(collisions)
    if max(deliveries) > max_d:
        max_d = max(deliveries)
    elif min(deliveries) < min_d:
        min_d = min(deliveries)

    if STATS:
        print("--- Episode Finished ---")
        print(f"Episode rewards: {total_reward} | Min: {min(episode_reward)} | Max: {max(episode_reward)}")
        print(f"Total Collisions: {total_collisions} | Min: {min(collisions)} | Max: {max(collisions)}")
        print(f"Total Deliveries: {total_deliveries} | Min: {min(deliveries)} | Max: {max(deliveries)}")
        print(" --- ")
        print(info)
        print(" --- ")

    if RECORD_OUTLIERS and total_reward < 0:
        save_animation(frames, f"videos/test/{model_name}_wp/video_ep{ep+1}")

print(f"--- Averages Over {EPISODES} Episodes ---")
print(f"Average rewards: {sum(all_rewards) / EPISODES} | Min: {min(all_rewards)} | Max: {max(all_rewards)}")
print(f"Average collisions: {sum(all_collisions) / EPISODES} | Min: {min(all_collisions)} | Max: {max(all_collisions)}")
print(f"Average deliveries: {sum(all_deliveries) / EPISODES} | Min: {min(all_deliveries)} | Max: {max(all_deliveries)}")
print("---")
