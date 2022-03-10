# NOTE: This was copied from my Lab6

import copy

# our standard imports
import numpy as np
import pandas as pd

# of course we need to be able to split into training and test
from sklearn.model_selection import train_test_split

# This is where we can get our models
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report


def scale(df):
    X = StandardScaler().fit(df).transform(df)
    X = pd.DataFrame(X, columns=df.columns)
    return X


def pca(X, random_state=42):
    columns = ["PC1 ", "PC2 "]
    X_pca = None
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    columns = ["PC1 (%0.2f)" % pca.explained_variance_ratio_[0], "PC2 (%0.2f)" % pca.explained_variance_ratio_[1]]
    X_pca = pd.DataFrame(X_pca, columns=columns)
    return X_pca


def kmeans(X, range_n_clusters=[2, 3, 4, 5, 6], random_state=10):
    kmeans_models = {}
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusterer.fit(X)
        kmeans_models[n_clusters] = clusterer
    return kmeans_models


def assign_labels(X, kmeans_models):
    cluster_labels = {}
    for n_clusters in kmeans_models.keys():
        cluster_labels[n_clusters] = kmeans_models[n_clusters].predict(X)
    cluster_labels = pd.DataFrame(cluster_labels)
    return cluster_labels


def silhouette_scores(X, cluster_labels):
    # NOTE: here's how you do this using sklearn:
    #   from sklearn.metrics import silhouette_score
    #   silhouette_avg = silhouette_score(X, cluster_labels)

    def d(x, y):  # For ease of use if you want it (distance between two points)
        return np.sqrt(np.sum((x - y) ** 2))

    s = np.zeros((len(X),))

    for index, point_i in X.iterrows():
        cluster_label = cluster_labels[index]
        Ci = X.loc[cluster_labels == cluster_label]

        # NOTE:
        # These were the one-liners that I wrote previously before breaking them up into nested for loops:
        # ai = sum([d(point, other) for j, other in Ci.iterrows()]) / (count - 1)  # (we subtract 1 because we don't included d(i, i) in the mean... d(i, i) is included in the sum but it just equals 0)
        # bi = min([sum(d(point, other) for j, other in Ck.iterrows()) / len(Ck) for Ck in [X.loc[cluster_labels == other_cluster_label] for other_cluster_label in cluster_labels.unique() if not other_cluster_label == cluster_label]])

        # calculate the distances between our specific point and the other points in the same cluster:
        internal_distances = []
        for index_other, other in Ci.iterrows():
            internal_distances.append(d(point_i, other))
        ai = sum(internal_distances) / (
                    len(Ci) - 1)  # (we subtract 1 because we don't included d(i, i) in the mean... d(i, i) is included in the sum but it just equals 0)

        avg_other_cluster_distances = []
        for other_cluster_label in cluster_labels.unique():  # go through all the clusters
            if not other_cluster_label == cluster_label:  # ignore its own cluster (we only want to iterate over the other clusters)
                for Ck in [X.loc[
                               cluster_labels == other_cluster_label]]:  # iterate over the other clusters (we grab only the rows in the specific other_cluster)
                    distances_to_other_points = [d(point_i, other) for _, other in
                                                 Ck.iterrows()]  # calcualte the distances between point and all the points in the other cluster
                    avg_distance = sum(distances_to_other_points) / len(
                        distances_to_other_points)  # average the distances
                    avg_other_cluster_distances.append(avg_distance)
        bi = min(avg_other_cluster_distances)

        if ai < bi:
            si = 1 - ai / bi
        elif ai == bi:
            si = 0
        else:
            si = bi / ai - 1

        s[index] = si

    return s


def bin_x(x, n_clusters=3, random_state=10):
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state).fit(x)
    return clusterer