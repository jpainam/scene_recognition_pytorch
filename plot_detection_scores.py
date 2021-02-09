import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == "__main__":
    train_data = pickle.load(open("data/annotations/mit_annotations.pkl", 'rb'))
    print(train_data.keys())
    images = train_data['images']
    categories = np.copy(train_data['categories'])
    attributes = np.copy(train_data['attributes'])
    labels = np.copy(train_data['labels'])

    min_score = np.ones(len(attributes))
    print(labels.shape)
    for label in labels:
        for idx in range(len(label)):
            if label[idx] != 0 and label[idx] < min_score[idx]:
                min_score[idx] = label[idx]

    r, l = 0, 128
    min_score = np.sort(min_score)[::-1]
    min_score = min_score[r:l]
    max_score = np.sort(np.max(labels, axis=0))[::-1]
    max_score = max_score[r:l]
    # assert len(attributes) == len(max_score) == len(min_score)
    x = np.arange(0, len(max_score))
    avg_score = np.mean([max_score, min_score], axis=0)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax2.plot(x, max_score, label="Maximum", color='red', marker='o', markevery=4,
             markeredgecolor='red', markerfacecolor='none')

    ax2.plot(x, avg_score, label="Average", color="blue", marker='o', markevery=4,
             markeredgecolor='blue', markerfacecolor='none')

    ax2.plot(x, min_score, label="Minimum", color='green',  marker='o', markevery=4,
             markeredgecolor='green', markerfacecolor='none')

    leg = ax2.legend()
    leg.get_frame().set_linewidth(0.0)
    ax2.set_xlabel('(b) $i^{th}$ Attribute', fontsize=12)
    ax2.set_ylabel('Detection scores', fontsize=12)
    ax2.set_axisbelow(True)
    ax2.grid()


    arr = np.count_nonzero(labels, axis=-1)
    #arr = arr[:10]
    count, bins, ignored = ax1.hist(arr, 20, facecolor='c', alpha=0.5, histtype='stepfilled', density=True)
    ax1.set_ylabel('Number of samples',  fontsize=12)
    ax1.set_xlabel(' (a) Detected attributes',  fontsize=12)
    from scipy.stats import norm

    mu, sigma = norm.fit(arr)
    #plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
    #         np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3)
    best_fit_line = norm.pdf(bins, mu, sigma)
    ax1.plot(bins, best_fit_line, linewidth=3)
    t = [str(int(l)) for l in np.linspace(0, 900, 9)]
    ax1.set_yticklabels(t)
    ax1.set_axisbelow(True)
    ax1.grid()
    plt.show()