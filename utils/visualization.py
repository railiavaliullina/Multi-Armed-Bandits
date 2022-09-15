import matplotlib.pyplot as plt
import numpy as np


def plot_hist(cfg, x, name, title):
    plt.hist(np.asarray(x), bins=cfg.arms_num, range=(0, cfg.arms_num))
    plt.title(title)
    plt.savefig(cfg.plots_dir + f'{name}.jpg')
    plt.cla()
    plt.close()
