import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime

def imshow(img, output_dir="./logs"):
    if torch.is_tensor(img):
        img = img.mumpy()

    img = img[np.newaxis, :]
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribution.png")


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)