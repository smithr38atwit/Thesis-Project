import gym
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from stable_baselines3 import A2C

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


env = gym.make("CartPole-v1")

model = A2C("MlpPolicy", env).learn(total_timesteps=10_000)

vec_env = model.get_env()
vec_env.render_mode = "rgb_array"
obs = vec_env.reset()
frames = [vec_env.render(mode="rgb_array")]
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    frames.append(vec_env.render(mode="rgb_array"))

anim = plot_animation(frames)
FFwriter = animation.FFMpegWriter(fps=10)
anim.save("test.mp4", writer=FFwriter)
