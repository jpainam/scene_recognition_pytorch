import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import itertools

val = np.load('data/annotations/val_feat.npz')
train = np.load('data/annotations/train_feat.npz')
feat1, labels1 = train['arr_0'], train['arr_1']
feat2, labels2 = val['arr_0'], val['arr_1']
uniq_labels = np.unique(labels1)

fig = plt.figure(figsize=(5, 2))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

marker = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V']
marker = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']

mks = itertools.cycle(['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V'])
markers = [next(mks) for i in range(len(uniq_labels))]


def plot_scatter(ax, my_data, my_y):
    sns.scatterplot(x="x", y="y",
                    hue="labels",
                    palette=sns.color_palette("hls", len(np.unique(my_y))),
                    data=my_data,
                    legend=False,
                    alpha=1.0,
                    linewidth=0,
                    markers=markers,
                    s=10,
                    ax=ax)


X, Y = [], []
selected_labels = np.random.choice(uniq_labels, 10)
for xx, yy in zip(feat2, labels2):
    if yy in selected_labels:
        X.append(xx)
        Y.append(yy)
embedded = TSNE(n_components=2, verbose=1, n_iter=1000).fit_transform(X, Y)
np.savez('logs/data1.npz', embedded, Y)
data1 = {'x': embedded[:, 0],
        'labels': Y,
        'y': embedded[:, 1]}
plot_scatter(ax1, my_data=data1, my_y=Y)

#==============================================
X, Y = [], []
selected_labels = np.random.choice(uniq_labels, 10)
for xx, yy in zip(feat2, labels2):
    if yy in selected_labels:
        X.append(xx)
        Y.append(yy)
embedded = TSNE(n_components=2, verbose=1, n_iter=1000).fit_transform(X, Y)
np.savez('logs/data2.npz', embedded, Y)
data2 = {'x': embedded[:, 0],
         'labels': Y,
         'y': embedded[:, 1]}
plot_scatter(ax2, my_data=data2, my_y=Y)

#===============================================
X, Y = [], []
selected_labels = np.random.choice(uniq_labels, 10)
for xx, yy in zip(feat2, labels2):
    if yy in selected_labels:
        X.append(xx)
        Y.append(yy)
embedded = TSNE(n_components=2, verbose=1, n_iter=1000).fit_transform(X, Y)
np.savez('logs/data3.npz', embedded, Y)
data3 = {'x': embedded[:, 0],
         'labels': Y,
         'y': embedded[:, 1]}
plot_scatter(ax3, my_data=data3, my_y=Y)

ax1.set_xlabel('Attributes Only Space', fontdict={'fontsize': 9})
ax1.set_ylabel('')
ax2.set_xlabel('Identities Only Space', fontdict={'fontsize': 9})
ax2.set_ylabel('')
ax3.set_xlabel('MASR (Ours)', fontdict={'fontsize': 9})
ax3.set_ylabel('')

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])
plt.tight_layout()
plt.show()
