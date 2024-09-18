import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# To get smooth animations
mpl.rc("animation", html="jshtml")


def _update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return (patch,)


def plot_animation(frames, repeat=False, interval=100):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")
    anim = animation.FuncAnimation(
        fig, _update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval
    )
    return anim, fig


def save_animation(frames, filename):
    anim, fig = plot_animation(frames)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(filename, writer=FFwriter)
    plt.close(fig)
    plt.clf()
    plt.close("all")
