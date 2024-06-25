import gym
import rware

env: gym.Env = gym.make("rware-small-4ag-v1")
for _ in range(1000000):
    env.reset()
    env.render()
    while True:
        actions = env.action_space.sample()
        n_obs, reward, done, info = env.step(actions)
        # env.render()
        if all(done):
            break
