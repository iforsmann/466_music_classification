# Implements a K Clustering from my Lab6 on our dataset

import pandas as pd
from sklearn.metrics import silhouette_score
from dataset_prep import get_dataframes
import Lab6_helper

if __name__ == "__main__":
    X_train, X_test, t_train, t_test = get_dataframes()

    kmeans_models = Lab6_helper.kmeans(X_train, range_n_clusters=[5, 9, 11, 13], random_state=10)
    cluster_labels = Lab6_helper.assign_labels(X_train, kmeans_models)

    # TODO: Look at a single cluster. How much variation in the music genre is there in that specific cluster. (Use a bar chart to display)

    # TODO: When calculating the "accuracy" you don't know which clusters correspond to which classes.
    #   An option would be to calculate the accuracy for every possible combination and then take the max.
    # print(cluster_labels[11].value_counts())
    # print(t_train.value_counts())

    s_df = pd.DataFrame(index=X_train.index, columns=cluster_labels.columns)
    for k in s_df.columns:
        s_df.loc[:, k] = silhouette_score(X_train, cluster_labels[k])
        # From scratch: Lab6_helper.silhouette_scores(X,cluster_labels[k])
            # but that ^ takes a LONG time

    s_df.index.name = "i"
    s_df = s_df.reset_index()

    source = s_df.melt(id_vars=["i"])
    source.columns = ["i", "k", "s"]

    print((s_df.mean(axis=0)))