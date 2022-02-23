# NOTE: This file is a copy from my Bayesian Mastery Checkpoint.

# NOTE:
#   This started as a copy of the Lab2_helper.py file.
#   I change the probabilities to gaussian but I didn't clean up the code so it's messy and inefficient.

import numpy as np
import pandas as pd
import math

# helper func
def find_priors(y):
    # y is a pd series object
    # value_counts finds the frequency of every unique element and divides by the total number of elements
    return y.value_counts(normalize=True, sort=False)


def compute_priors(y, uniform=False):
    priors = {}
    uniform_prior = 1 / len(y.unique())
    for index, prior in find_priors(y).items():
        priors[f"{y.name}={index}"] = (uniform_prior if uniform else prior)
    return priors


# helper func
def find_cond_probs(x, y):
    """This function finds all the conditional probabilities
            for each value in class x assuming each value in class y.
    Ex: A single probability would be: Pr(Sex==female|Survived==0)
        x would be the Sex column (a series object)
        y would be the Survived column (a series object)
        conditional_probs is all the probabilities but you can select just one:
            you can then select female and 0 by: conditional_probs[("female", 0)]
    """
    # I found a "shortcut" way to do this but there's a more manual example in Chapter 2 notes.
    # The shortcut calculation: https://stackoverflow.com/a/37818394/11866492
    # priors = titanic_df.groupby('Survived').size().div(len(titanic_df))
    # titanic_df.groupby(['Sex', 'Survived']).size().div(len(titanic_df)).div(priors, axis=0, level='Survived')

    # I improved upon the shortcut:
    # priors = titanic_df["Survived"].value_counts(normalize=True, sort=False)
    # titanic_df.value_counts(subset=['Sex', 'Survived'], normalize=True, sort=False).div(priors, axis=0, level='Survived')

    # I adapted my improvement to this function:
    priors = find_priors(y)
    subset = pd.concat([x, y], axis=1)  # since we can't access the whole dataframe, we need to concatenate the series objects we're given
    conditional_probs = subset.value_counts(normalize=True, sort=False).div(priors, axis=0, level=y.name)

    return conditional_probs


def specific_class_conditional(x,xv,y,yv):
    return find_cond_probs(x, y)[(xv, yv)]  # conditional_probs is a series and we're indexing by (xv, yv) (Ex: ('female', 0))


def class_conditional(X,y):
    probs = {}
    for df_class in X:
        # add all conditional probabilities of Pr(df_class|y.name) to probs:
        for index, prob in find_cond_probs(X[df_class], y).items():
            class_val = index[0]
            assumed_class = y.name
            assumed_val = index[1]
            probs[f"{df_class}={class_val}|{assumed_class}={assumed_val}"] = prob


    # Add a probability of 0.0 for any combination that didn't have a probability already.
    # We had to do this because the answer in Exercise 3 included a probability of 0.0 for
    #     Age=70|Survived=1 and Age=80|Survived=0
    # These two combinations were not in the data set so probabilities were not generated.
    for df_class in X:
        for class_val in X[df_class].unique():
            for assumed_val in y.unique():
                assumed_class = y.name
                if f"{df_class}={class_val}|{assumed_class}={assumed_val}" not in probs:
                    probs[f"{df_class}={class_val}|{assumed_class}={assumed_val}"] = 0.0

    return probs


# calculates the probability of NOT something
def posterior_helper(df_class, class_val, assumed, probs):
    not_prob = 0

    for key, prob in probs.items():
        if f"{df_class}={class_val}" in key and key != f"{df_class}={class_val}|{assumed}":
            not_prob += prob

    return not_prob


def gaussian_prob(x, mean, stdev):
    # this is the gaussian (normal) distribution equation:
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# posterior probabilities
def posteriors(priors, x,  means, std_devs):
    # NOTES:
    # numer = Pr(Survived=0 (the prior)) * Pr(Pclass=3|Survived=0) * Pr(Sex=female|Survived=0) * Pr(Age=20|Survived=0)
    # denom = numer
    # denom += Pr(Survived= not 0 (sum of all the other priors)) * Pr(Pclass=3|Survived= not 0) * Pr(Sex=female|Survived= not 0) * Pr(Age=20|Survived= not 0)
    #        = Pr(Survived=1) * Pr(Pclass=3|Survived=1) * Pr(Sex=female|Survived=1) * Pr(Age=20|Survived=1)

    post_probs = {}

    for index, prior in priors.items():
        species = index.split('=')[1]
        numerator = prior
        cur_key = f"{index}|"
        for df_class, value in x.items():
            try:
                # numerator *= probs[f"{df_class}={value}|{index}"]
                numerator *= gaussian_prob(value, means.at[species, df_class], std_devs.at[species, df_class])  # TESTING
                cur_key += f"{df_class}={value},"
            except KeyError as e:
                # print(f"KeyError: {e}")
                pass
        cur_key = cur_key[:-1]  # remove the last comma

        post_probs[cur_key] = numerator  # TESTING

        # if numerator == 0:
        #     post_probs[cur_key] = 0.0
        # else:
        #     denom = numerator
        #     denom_not_prob = 0.0
        #     for index2, prior2 in priors.items():
        #         if index2 != index:
        #             denom_not_prob += prior2
        #     for df_class, class_val in x.items():
        #         denom_not_prob *= posterior_helper(df_class, class_val, index, probs)
        #     denom += denom_not_prob
        #
        #     post_probs[cur_key] = numerator / denom

    return post_probs


# exercise 5
def train_test_split(X,y,test_frac=0.5):
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs,:]
    y = y.iloc[inxs]

    first_set = math.ceil(len(y) * test_frac)
    last_set = math.floor(len(y) * test_frac)
    Xtrain, ytrain = X.head(first_set), y.head(first_set)
    Xtest, ytest = X.tail(last_set), y.tail(last_set)
    return Xtrain,ytrain,Xtest,ytest


def exercise_6_helper(priors, Xtest, ytest, means, std_devs):
    correct = 0
    total = 0
    for index, x in Xtest.iterrows():
        post_probs = posteriors(priors, x, means, std_devs)
        prediction = max(post_probs, key=post_probs.get)  # gets the key of the highest probability
        pred_val = prediction.split('|')[0].split('=')[1]  # get the predicted value from "Survived=0|Pclass=3,Sex=male,Age=20"

        if pred_val == str(ytest.loc[index]):
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def get_means_and_std_devs(Xtrain, ytrain):
    train_df = pd.concat([Xtrain, ytrain], axis=1).groupby(ytrain.name)
    return train_df.mean(), train_df.std()


def exercise_6(Xtrain, ytrain, Xtest, ytest):
    # probs = class_conditional(Xtrain, ytrain)
    priors = compute_priors(ytrain)

    means, std_devs = get_means_and_std_devs(Xtrain, ytrain)

    return exercise_6_helper(priors, Xtest, ytest, means, std_devs)


def exercise_7(Xtrain, ytrain, Xtest, ytest, npermutations=10, exercise8=False):
    # find the original accuracy
    # probs = class_conditional(Xtrain, ytrain)
    priors = compute_priors(ytrain)
    means, std_devs = get_means_and_std_devs(Xtrain, ytrain)
    orig_accuracy = exercise_6_helper(priors, Xtest, ytest, means, std_devs)

    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0

    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            if exercise8:
                Xtrain2 = Xtrain.copy()
                Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values

                test_df = Xtest

                # we need to recalculate the probs and priors because we're permuting the TRAINING data
                # probs = class_conditional(Xtrain2, ytrain)
                priors = compute_priors(ytrain)
            else:
                Xtest2 = Xtest.copy()
                Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values

                test_df = Xtest2

            accuracy = exercise_6_helper(priors, test_df, ytest, means, std_devs)
            importances[col] += accuracy


        importances[col] /= npermutations


    for key, accuracy in importances.items():
        importances[key] = orig_accuracy - accuracy

    return importances


def exercise_8(Xtrain, ytrain, Xtest, ytest, npermutations=20):
    return exercise_7(Xtrain, ytrain, Xtest, ytest, npermutations=npermutations, exercise8=True)


