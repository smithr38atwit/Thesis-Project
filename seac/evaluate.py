import gym
import rware
import torch
from a2c import A2C
from wrappers import Monitor, RecordEpisodeStatistics, TimeLimit

model_name = "base_40k_2"
path = f"C:/Users/smithr38/Code/School/Thesis-Project/models/{model_name}"
env_name = "rware-tiny-2ag-v1"
time_limit = 500  # 25 for LBF

EPISODES = 5
SHOW = True

env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

total_infos = {}
for ep in range(EPISODES):
    env = gym.make(env_name)
    if SHOW:
        env = Monitor(env, f"videos/test/{model_name}_wp/video_ep{ep+1}", mode="evaluation")
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)

    obs = env.reset()
    done = [False] * len(agents)

    while not all(done):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _, _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        if SHOW:
            env.render()
        obs, _, done, info = env.step(actions)
    obs = env.reset()
    total_infos
    if SHOW:
        print("--- Episode Finished ---")
        print(f"Episode rewards: {sum(info['episode_reward'])}")
        print(f"Total Collisions: {sum(info['collisions'])}")
        print(info)
        print(" --- ")
