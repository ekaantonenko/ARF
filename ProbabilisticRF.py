import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from DataGenerator import generate_individual_datasets
from sklearn import tree
import matplotlib.pyplot as plt


class ProbabilisticTreeForest:

    def __init__(self, cl_type, n_base_trees, criterion, max_depth, min_impurity_decrease,
                 min_samples_split, max_features, dimension=None):
        """
        Initializes the Probabilistic Tree Forest classifier. It is either based on Random Forests or in Extra Trees
        :param data: The given data to learn the initial model
        :param cl_type: Either 'RF' for Random forests or 'ET' for Extra Trees or 'DT' for single Decision Tree
        :param n_base_trees: Number of base trees that conform the ensemble
        :param criterion: Impurity driven learning criteria 'gini' for Gini, 'entropy' for Shannon entropy or 'log_loss'
        :param max_depth: Max depth of the trees for pruning
        :param min_samples_split: Min number of samples to consider to make a split
        :param max_features: Max number of features to be considered in a subset to learn a Tree
        """
        self.dimension = dimension  # This is fitted when calling dummyfit or now
        self.cl_type = cl_type
        if cl_type == "RF":
            self.classifier = RandomForestClassifier(n_base_trees, criterion=criterion, max_depth=max_depth,
                                                     min_impurity_decrease=min_impurity_decrease,
                                                     min_samples_split=min_samples_split, max_features=max_features)
        elif cl_type == "ET":
            self.classifier = ExtraTreesClassifier(n_base_trees, criterion=criterion, max_depth=max_depth,
                                                   min_impurity_decrease=min_impurity_decrease,
                                                   min_samples_split=min_samples_split, max_features=max_features)
        elif cl_type == "DT":
            self.classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                     min_impurity_decrease=min_impurity_decrease,
                                                     min_samples_split=min_samples_split, max_features=max_features)

    def dummy_fit(self, dimension, n_variables_X, n_samples=1000):
        self.dimension = dimension
        X = np.array(list(range(dimension[0])) * n_samples).reshape(-1, 1)[:n_samples]
        for i in range(1, n_variables_X):
            to_stack = np.array(list(range(dimension[i])) * n_samples).reshape(-1, 1)[:n_samples]
            X = np.hstack([X, to_stack])
        self.classifier.fit(X, X)

    def relearn_individual_trees(self, X, Y, sample_weights, sample=False):
        """
        Relearns the base classifiers with individually sampled data for each tree.
        The classifier needs to be fitted first.
        :param Y: if sample == True probabilistic data to be used, the output of predict_proba. If not, regular labels
        :param sample_weights: sample weights to be used in the fitting
        :param sample: Whether you would like to sample an individual dataset or not for each tree, default=False
        """
        if sample:
            # The Y contains a preds_proba so we first extract the labels
            y_test_unique = np.argmax(Y, axis=2).T
        else:
            y_test_unique = Y
        if self.cl_type == 'DT':
            if sample:
                y_test_unique = generate_individual_datasets(Y)
            self.classifier.fit(X, y_test_unique, sample_weights)
        else:
            self.classifier.fit(X, y_test_unique, sample_weights)
            # for e in self.classifier.estimators_:
                # fit the data used for each tree
                # if sample:
                #     y_test_unique = generate_individual_datasets(Y)
                # e.fit(X[:, e.feature_importances_ == 0], y_test_unique[:, e.feature_importances_ == 0], sample_weights)

    def predict_and_correct(self, X, Y, na_value):
        """
        Predicts the probabilities for the given X
        :param X: A full subset of instances with all the values
        :param Y: The variables to be predicted and the ones that are not missing so the predictions are corrected.
        :return: A 3D matrix containing the predictions (#instance, variable Y, dimension) and the sample weights
        for each tree. These weights are the individual probabilities assigned by each tree to the instances
        """
        # Predict the missing values
        preds_proba = self.classifier.predict_proba(X)
        # Correct the needed ones
        preds_proba = self.correct_predictions(preds_proba, Y, na_value)
        return preds_proba

    def correct_predictions(self, preds_proba, Y, na_value=-1):
        # Reshape if necessary
        for ix, d in enumerate(self.dimension):
            missing_columns = d - preds_proba[ix].shape[1]
            if missing_columns > 0:
                preds_proba[ix] = np.hstack([preds_proba[ix], np.zeros((preds_proba[ix].shape[0], missing_columns))])

        for ixr, r in enumerate(Y):  # Per row, instance
            for ixc, c in enumerate(r):  # Per column, variable
                if c != na_value:
                    to_insert = [0] * self.dimension[ixc]
                    to_insert[int(c)] = 1
                    preds_proba[ixc][ixr, :] = to_insert  # Alter the preds because this should not be predicted
        return preds_proba

