import glob
import os

import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


########################
###### Animations ######
########################

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# To get smooth animations
mpl.rc("animation", html="jshtml")


def _update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return (patch,)


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")
    anim = animation.FuncAnimation(
        fig, _update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval
    )
    plt.close()
    return anim


def save_animation(frames, filename):
    anim = plot_animation(frames)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save("videos\\" + filename, writer=FFwriter)
