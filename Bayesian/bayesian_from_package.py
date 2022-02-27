# Implements a Gaussian Naive Bayesian classifier from sklearn on our dataset

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pickle

from dataset_prep import get_dataframes


def my_train(X, t):
    model = GaussianNB()
    model.fit(X, t)
    return model

def create_model():
    model = my_train(X_train, t_train)

    # save the model to disk
    pickle.dump(model, open(model_filename, 'wb'))


def test_model():
    # load the model from disk
    model = pickle.load(open(model_filename, 'rb'))

    print(f"accuracy: {model.score(X_test, t_test)}")
    # model.score(X_test, t_test) just uses model.predict(X_test) and compares the predicted values to the target values (t_test)

    print(classification_report(t_test, model.predict(X_test)))
    # NOTE: Currently classes {1, 5} are not being predicted by our model (but they are in the target values).


if __name__ == "__main__":
    X_train, X_test, t_train, t_test = get_dataframes()

    model_filename = 'bayesian_model.sav'

    create_model()
    test_model()


"""
accuracy: 0.36013542107490476

              precision    recall  f1-score   support

           0       0.42      0.54      0.47        78
           1       0.00      0.00      0.00       206
           2       0.39      0.04      0.07       193
           3       0.54      0.54      0.54        52
           4       0.25      0.35      0.29        26
           5       0.00      0.00      0.00       100
           6       0.26      0.21      0.23       393
           7       0.80      0.88      0.84        99
           8       0.42      0.09      0.14       285
           9       0.52      0.06      0.11       249
          10       0.34      0.81      0.48       682

    accuracy                           0.36      2363
   macro avg       0.36      0.32      0.29      2363
weighted avg       0.34      0.36      0.28      2363
"""
