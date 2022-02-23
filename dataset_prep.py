import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataframes():
    train = pd.read_csv("music_genre_dataset/train.csv")

    train = train.drop(["Artist Name", "Track Name"], axis=1)  # drop categorical columns
    train = train.dropna()  # if any row has NaN in any cell, the whole row is dropped (indices remain the same)
    # TODO: normalize train?

    X = train.drop(["Class"], axis=1)
    y = train["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

