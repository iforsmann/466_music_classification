# Implements a Gaussian Naive Bayesian classifier from my Lab2 on our dataset

import mastery_helper
from dataset_prep import get_dataframes


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_dataframes()

    # accuracy = mastery_helper.exercise_6(X_train, y_train, X_test, y_test)
    # print(accuracy)

    # Xtest = Xtest.applymap(lambda x: x + random.random() / 50.0)  # add some randomness to the test data so we can see if the normal distribution is working
    # accuracy = mastery_helper.exercise_6(Xtrain, ytrain, Xtest, ytest)
    # print(accuracy)

    importances = mastery_helper.exercise_7(X_train, y_train, X_test, y_test)
    print(importances)


# accuracy = 0.28861616589081673
# importances = {'Popularity': -5.551115123125783e-17, 'danceability': -5.551115123125783e-17, 'energy': -5.551115123125783e-17, 'key': -5.551115123125783e-17, 'loudness': -5.551115123125783e-17, 'mode': -5.551115123125783e-17, 'speechiness': -5.551115123125783e-17, 'acousticness': -5.551115123125783e-17, 'instrumentalness': -5.551115123125783e-17, 'liveness': -5.551115123125783e-17, 'valence': -5.551115123125783e-17, 'tempo': -5.551115123125783e-17, 'duration_in min/ms': -5.551115123125783e-17, 'time_signature': -5.551115123125783e-17}
