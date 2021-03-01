import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#sns.set(font_scale=1.4)
import numpy as np
import pickle
import os


def get_correlation(annotations):
    train_data = pickle.load(open(os.path.join("data/annotations/", annotations), 'rb'))

    images = train_data['images']
    categories = np.copy(train_data['categories'])
    attributes = np.copy(train_data['attributes'])
    labels = np.copy(train_data['labels'])

    c_labels = np.copy(train_data['labels'])
    sorted_index = np.argsort(np.sum(c_labels, axis=0))[::-1]
    sorted_index = sorted_index[:10]
    correlations = c_labels[:, sorted_index]
    return correlations, attributes[sorted_index]


if __name__ == "__main__":
    sun_corr, sun_att = get_correlation("sun_annotations.pkl")
    #correlations2, attributes2 = get_correlation("ade_annotations.pkl")
    mit_cor, mit_att = get_correlation("mit_annotations.pkl")
    #place_cor, place_att = get_correlation("place_annotations.pkl")
    #print(correlations.shape)
    sun_df = pd.DataFrame(sun_corr, columns=sun_att)
    #df2 = pd.DataFrame(correlations2, columns=attributes2)

    mit_df = pd.DataFrame(mit_cor, columns=mit_att)
    #place_df = pd.DataFrame(place_cor, columns=place_att)

    sun_corrMatrix = sun_df.corr()
    mit_corrMatrix = mit_df.corr()
    #mit_corrMatrix = mit_df.corr()
    #place_corrMatrix = place_df.corr()
    #print(corrMatrix)

    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (5,2))
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (5,2))
    g = sns.heatmap(mit_corrMatrix, annot=True, cmap='Greys', fmt=".1f", linewidths=.5, cbar = False, ax = ax1)
    g = sns.heatmap(sun_corrMatrix, annot=True, cmap='Greys', fmt=".1f", linewidths=.5, cbar = False, ax = ax2)

    #g = sns.heatmap(mit_corrMatrix, annot=True, cmap='Greys', fmt=".1f", linewidths=.5, cbar = False, ax = ax3)
    #g = sns.heatmap(place_corrMatrix, annot=True, cmap='Greys', fmt=".1f", linewidths=.5, cbar = False, ax = ax4)

    ax1.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    ax2.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

    #ax3.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    #ax4.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    # annot_kws={"size": 20}
    #axes[0].set_yticks(rotation=0)
    plt.setp(ax1.get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=90, horizontalalignment='right')
    #plt.setp(ax3.get_xticklabels(), rotation=90, horizontalalignment='right')
    #plt.setp(ax4.get_xticklabels(), rotation=90, horizontalalignment='right')

    ax1.set_xlabel("(a) MIT67")
    #axes[1].set_xlabel("(b) MIT67")
    ax2.set_xlabel("(b) SUN397")
    #ax3.set_xlabel("(c) MIT")
    #ax4.set_xlabel("(d) Places")
    #axes[0].xaxis.tick_top() # x axis on top
    #axes[0].xaxis.set_label_position('top')
    #plt.savefig('heatmap.pdf')
    plt.show()