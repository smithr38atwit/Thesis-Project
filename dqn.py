import gym
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

import rware

# For animations

mpl.rc("animation", html="jshtml")


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return (patch,)


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval
    )
    plt.close()
    return anim


env: gym.Env = gym.make("rware-tiny-2ag-v1")
env.reset()

frames = [env.render(mode="rgb_array")]
for _ in range(10):
    actions = env.action_space.sample()
    n_obs, reward, done, info = env.step(actions)
    frames.append(env.render(mode="rgb_array"))
env.close()

anim = plot_animation(frames)
anim.save("test.gif")
