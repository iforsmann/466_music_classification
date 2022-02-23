# Implements a Gaussian Naive Bayesian classifier from sklearn on our dataset

from sklearn.naive_bayes import GaussianNB
import pickle

from dataset_prep import get_dataframes


def my_train(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model

def create_model():
    model = my_train(X_train, y_train)

    # save the model to disk
    pickle.dump(model, open(model_filename, 'wb'))


def test_model():
    # load the model from disk
    model = pickle.load(open(model_filename, 'rb'))

    print(model.score(X_test, y_test))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_dataframes()

    model_filename = 'bayesian_model.sav'

    create_model()
    test_model()


# accuracy = 0.36013542107490476
