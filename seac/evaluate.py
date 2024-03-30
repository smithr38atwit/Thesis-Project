import gym
import torch
from a2c import A2C
from wrappers import Monitor, RecordEpisodeStatistics, TimeLimit

import rware

model_name = "model_no_people"
path = f"/home/lambda10/rsmith_thesis/Thesis-Project/models/{model_name}"
env_name = "rware-tiny-2ag-v1"
time_limit = 500  # 25 for LBF

EPISODES = 5

env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

for ep in range(EPISODES):
    env = gym.make(env_name)
    env = Monitor(env, f"videos/{model_name}/video_ep{ep+1}", mode="evaluation")
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)

    obs = env.reset()
    done = [False] * len(agents)

    while not all(done):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _, _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        env.render()
        obs, _, done, info = env.step(actions)
    obs = env.reset()
    print("--- Episode Finished ---")
    print(f"Episode rewards: {sum(info['episode_reward'])}")
    print(info)
    print(" --- ")
