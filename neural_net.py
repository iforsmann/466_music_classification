from tkinter.tix import Tree
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math

test_size = 0.3
val_size = 0.3
NUM_EPOCHS = 30
BATCH_SIZE = 10
NUM_NEURONS = 32
ALPHA = 0.00001


def main():
    df = pd.read_csv("./data/train.csv") 
    df.drop("Artist Name", axis=1, inplace=True)
    df.drop("Track Name", axis=1, inplace=True)
    df.drop("time_signature", axis=1, inplace=True)
    df.drop("key", axis=1, inplace=True)
    df.drop("mode", axis=1, inplace=True)
    df.dropna(inplace=True)
    targets = df["Class"]
    df.drop("Class", axis=1, inplace=True)
    music_df = (df - df.mean()) / df.std()
    music_df.reset_index(drop=True, inplace=True)
    targets.reset_index(drop=True, inplace=True)
    t = pd.get_dummies(targets)
    num_features = len(music_df.columns)
    X_train, X_test, t_train, t_test = train_test_split(music_df, t)
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    t_train_copy = t_train.copy()
    t_test_copy = t_test.copy()

    # k_means_cluster(music_df)

    print("Accuracy: " + str(scratch_model(X_train, X_test, t_train, t_test)))
    print("Permutation Feature Importance: " + str(feature_importance(X_train_copy, t_train_copy, X_test_copy, t_test_copy)))

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(NUM_NEURONS, input_dim=num_features, activation="relu"),
    #     tf.keras.layers.Dense(NUM_NEURONS, activation="relu"),
    #     tf.keras.layers.Dense(11, activation="softmax")
    # ])
    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #             metrics=['accuracy'])

    # model.fit(np.asarray(X_train), np.asarray(t_train), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    # y_test = model.predict(np.asarray(X_test))
    # t_test = np.asarray(t_test).astype('float32')
    # accuracy = accuracy_score(np.argmax(t_test, axis=1), np.argmax(y_test, axis=1))
    
    # print(accuracy)
    # print(t)

def activation_function(layer):
    if layer > 0:
        return layer
    else:
        return 0

def scratch_model(X_train, X_test, t_train, t_test):
    weights_0_1 = np.random.rand(NUM_NEURONS, len(X_train.columns)) * 2 - 1
    weights_1_2 = np.random.rand(NUM_NEURONS, NUM_NEURONS) * 2 - 1
    weights_2_3 = np.random.rand(11, NUM_NEURONS) * 2 - 1
    layer_1 = np.dot(weights_0_1, X_train.iloc[0])
    layer_2 = np.dot(weights_1_2, layer_1)
    output_layer = np.dot(weights_2_3, layer_2)
    vfunc = np.vectorize(activation)

    for epoch in range(NUM_EPOCHS):
        for i in range(len(X_train)):
            layer_0 = X_train.iloc[i]
            layer_1 = np.dot(weights_0_1, layer_0)
            #add activation function here
            # layer_1 = vfunc(layer_1)
            layer_2 = np.dot(weights_1_2, layer_1)
            # layer_2 = vfunc(layer_2)
            output_layer = np.dot(weights_2_3, layer_2)
            d3 = np.asarray(output_layer - t_train.iloc[i])
            # mask1 = np.where(layer_1 <= 0, 0, 1)
            # mask2 = np.where(layer_2 <= 0, 0, 1)
            d2 = np.dot(np.transpose(weights_2_3), d3)# * mask2
            d1 = np.dot(np.transpose(weights_1_2), d2)# * mask1
            update_2_3 = np.outer(d3, layer_2) * ALPHA
            update_1_2 = np.outer(d2, layer_1) * ALPHA
            update_0_1 = np.outer(d1, layer_0) * ALPHA
            weights_0_1 = weights_0_1 - update_0_1
            weights_1_2 = weights_1_2 - update_1_2
            weights_2_3 = weights_2_3 - update_2_3
        progress(epoch, NUM_EPOCHS)
    print()
    correct = 0
    for i in range(len(X_test)):
        layer_1 = np.dot(weights_0_1, X_test.iloc[i])
        layer_2 = np.dot(weights_1_2, layer_1)
        output_layer = np.dot(weights_2_3, layer_2)
        if np.argmax(output_layer) == np.argmax(t_test.iloc[i]):
            correct += 1

    return correct / len(X_test)

def k_means_cluster(data):
    scores = {}
    for i in range(2, 16):
        model = KMeans(n_clusters=i).fit_predict(np.asarray(data))
        scores[i] = silhouette_score(np.asarray(data), model)
    print(scores)


def feature_importance(Xtrain,ytrain,Xtest,ytest, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = scratch_model(Xtrain,Xtest,ytrain,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            new_accuracy = scratch_model(Xtrain,Xtest2,ytrain,ytest)
            importances[col] += (1-new_accuracy) - (1-orig_accuracy)
        importances[col] = importances[col]/npermutations
    return importances


def activation(val):
    return max(0, val)

def progress(i, total):
    print("\rProgress: " + str(math.ceil(100 * float(i) / total)), end="\r")

if __name__ == "__main__":
    main()