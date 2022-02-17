from tkinter.tix import Tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

test_size = 0.3
val_size = 0.3


def main():
    df = pd.read_csv("./data/train.csv") 
    df.drop("Artist Name", axis=1, inplace=True)
    df.drop("Track Name", axis=1, inplace=True)
    df.drop("time_signature", axis=1, inplace=True)
    df.drop("key", axis=1, inplace=True)
    df.drop("mode", axis=1, inplace=True)
    df.dropna(inplace=True)
    t = df["Class"]
    df.drop("Class", axis=1, inplace=True)
    music_df = (df - df.mean()) / df.std()
    music_df.reset_index(drop=True, inplace=True)
    t.reset_index(drop=True, inplace=True)
    print(music_df)
    print(t)

if __name__ == "__main__":
    main()