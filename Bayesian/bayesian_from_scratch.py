# Implements a Gaussian Naive Bayesian classifier from my Lab2 on our dataset

import random
import mastery_helper
from dataset_prep import get_dataframes


if __name__ == "__main__":
    X_train, X_test, t_train, t_test = get_dataframes()

    accuracy = mastery_helper.exercise_6(X_train, t_train, X_test, t_test)
    print(accuracy)

    # X_test = X_test.applymap(lambda x: x + random.random() / 5.0)  # add some randomness to the test data so we can see if the normal distribution is working
    # accuracy = mastery_helper.exercise_6(X_train, t_train, X_test, t_test)
    # print(accuracy)

    # NOTE: This takes a LONG time on the music genre dataset.
    # importances = mastery_helper.exercise_7(X_train, t_train, X_test, t_test)
    # print(importances)


"""
accuracy: 0.43419382141345747
importances: {'popularity': 0.015107913669064721, 'danceability': 0.02039779940753278, 'energy': 0.02767668218366487, 'key': -0.0011426153195090438, 'loudness': 0.00300465509944986, 'mode': 0.0016504443504020139, 'speechiness': 0.02628015234870923, 'acousticness': 0.05742699957680908, 'instrumentalness': 0.03728311468472284, 'liveness': 0.0018620397799407606, 'valence': 0.02175201015658068, 'tempo': -0.0034701650444350363, 'duration': 0.10710960643250106, 'time_signature': 0.004189589504866753}
"""
