import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataframes():
    target_col = "class"

    df = pd.read_csv("music_genre_dataset/train.csv", header=0)
    df = df.drop(columns=["artist_name", "track_name"])  # drop categorical columns
    df = df.dropna()  # if any row has NaN in any cell, the whole row is dropped (indices remain the same)
    df[target_col] = df[target_col].apply(lambda x: str(x))  # turn the class values into strings  (mastery_helper.py requires this)
    # TODO: normalize df?

    # print(df.head())

    X = df.drop([target_col], axis=1)
    t = df[target_col]

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=0)
    return X_train, X_test, t_train, t_test

