import gym
from stable_baselines3 import DQN

import rware
from utils.animations import save_animation

env: gym.Env = gym.make("rware-tiny-2ag-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000, log_interval=4)
model.save(r"models\dqn_test")

frames = [env.render(mode="rgb_array")]
obs = env.reset()
while True:
    actions, _states = model.predict()
    n_obs, reward, done, info = env.step(actions)
    frames.append(env.render(mode="rgb_array"))
env.close()

anim = save_animation(frames, "test.mp4")
