import matplotlib.pyplot as plt


def plot_environment(img, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    return img
