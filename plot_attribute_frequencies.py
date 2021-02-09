import numpy as np
import pickle
import os
import glob
import random
import matplotlib.pyplot as plt
import cv2
from matplotlib.ticker import ScalarFormatter


def  getinfo(annotations):
    train_data = pickle.load(open(annotations, 'rb'))
    print(train_data.keys())
    images = train_data['images']
    categories = np.copy(train_data['categories'])
    attributes = np.copy(train_data['attributes'])
    labels = np.copy(train_data['labels'])

    # annotated instances
    assert len(images) == len(labels)
    bars = np.zeros(len(attributes), dtype=np.int32)
    for i, (img, label) in enumerate(zip(images, labels)):
        label[label != 0] = 1.
        # label[label <= confidence_score] = 0
        bars = bars + label

    sorted_index = np.argsort(bars)[::-1]
    bars = bars[sorted_index]
    attributes = attributes[sorted_index]

    return bars, attributes


if __name__ == "__main__":


    def forward(x):
        return x ** (1 / 5)

    def inverse(x):
        return x ** 5


    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

    bars, attributes = getinfo('data/annotations/mit_annotations.pkl')
    bars = bars[:len(bars) - 6]
    attributes = attributes[:len(attributes) - 6]

    # orig_bars = bars.copy()
    # print(max(bars))
    # bars[bars < 100] = bars[bars < 100] * 100
    # bars[bars > 1000] = (bars[bars > 1000] - min(bars[bars > 1000])) / (max(bars) - min(bars[bars > 1000]))

    h = ax1.bar(range(len(bars)), bars, label=attributes, color='g')


    xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h]
    # plt.grid(axis='y')
    ax1.set_xticks(xticks_pos)
    ax1.set_xticklabels(attributes, rotation=90, fontsize=4)
    # transform x axis into logarithmic scale with base 2
    ax1.set_yscale("function", functions=(forward, inverse))

    ax1.yaxis.grid()
    # modify x axis ticks from exponential representation to float
    #ax1.get_yaxis().set_major_formatter(ScalarFormatter())
    plt.setp(ax1.get_yticklabels(), fontsize=8)
    # ax.set_ylim(bottom=0);
    ax1.set_xlim(left=-1, right=max(xticks_pos))

    barss, attributess = getinfo('data/annotations/sun_annotations.pkl')
    barss = barss[:len(barss) - 2]
    attributess = attributess[:len(attributess) - 2]
    h2 = ax2.bar(range(len(barss)), barss, label=attributess, color='c')
    xticks_pos2 = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h2]
    ax2.set_xticks(xticks_pos2)
    ax2.set_xticklabels(attributess, rotation=90, fontsize=4)
    ax2.set_yscale("function", functions=(forward, inverse))
    plt.setp(ax2.get_yticklabels(), fontsize=8)
    ax2.set_xlim(left=-1, right=max(xticks_pos2))

    ax1.set_title('(a) MIT67', fontdict={'fontsize': 12})
    ax2.set_title('(b) SUN397', fontdict={'fontsize': 12})
    plt.subplots_adjust(bottom=0.3)
    plt.show()