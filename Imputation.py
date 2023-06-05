import pandas as pd
import numpy as np
from sklearn import ensemble, neural_network
from scipy import stats
from sklearn.metrics import *
from sklearn.impute import SimpleImputer

import time

from sklearn.base import clone, BaseEstimator

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

import warnings

from ProbabilisticRF import ProbabilisticTreeForest

warnings.filterwarnings('ignore')

from numpy.random import RandomState as RS

###### Local packages ######
from local_metrics import accuracy_individual


class Imputation(BaseEstimator):
    """TBD
    
    Parameters
    ----------
    method :
    
    window_size :
    
    initial_imputation :
    
    ensemble :
    
    direction :
    
    n_iter : int, default=1
    
    na :
        
        
    Attributes
    ----------
    TBD
    
    References
    ----------
    TBD
    
    Examples
    --------
    TBD
    """
    
    def __init__(self, estimator, *, method='iterative', window_size=10, initial_imputation='random', n_ensemble=1,
        direction='left_right', n_iter=10, na=-1, return_time=False, dim=None):
        self.method = method
        self.delta = window_size
        self.initial_imputation = initial_imputation
        self.n_ensemble = n_ensemble
        self.direction = direction
        self.max_iter = n_iter
        self.na = na
        self.loglikelihood = []
        
        self.past = 0

        self.return_time = return_time
        self.estimator = estimator

        # Generic parameters from sklearn
        n_base_trees = 20
        criterion = 'gini'
        max_depth = None  # Until all are pure
        min_impurity_decrease = 0  # If the decrease is less than this, it won't split
        min_samples_split = 5
        max_features = None
        ditARF_cl_type = 'RF'
        
        if estimator == 'RF':
            self.base_estimator = ensemble.RandomForestClassifier(n_estimators=n_base_trees, max_features=max_features,
                                                                  min_samples_split=min_samples_split)
        elif estimator == 'AE':
            self.base_estimator = neural_network.MLPClassifier(hidden_layer_sizes=(10,), activation='logistic')
        elif estimator == 'PCA':
            self.base_estimator = neural_network.MLPClassifier(hidden_layer_sizes=(10,), activation='identity')
        elif estimator == 'ditARF':
            self.base_estimator = ProbabilisticTreeForest(ditARF_cl_type, n_base_trees, criterion, max_depth,
                                                          min_impurity_decrease, min_samples_split, max_features)
            # self.base_estimator.dummy_fit(dimension=2, n_variables_X=2)  # Dummy fitting with random numbers

    def _initial_imputation(self, X_na, dimension):
        """
        Initial imputation
        """
        N, p = X_na.shape[0], X_na.shape[1]
        X_imp = None  # Adds robustness to the method in case it does not enter in any ifs
        if self.initial_imputation == 'random':
            initialMask = np.zeros((N,0))
            for i in range(p):
                x_unique = np.unique(X_na[:,i])
                x_unique = x_unique[x_unique != self.na]
                mask_i = np.random.choice(x_unique, size=(N,1))
                initialMask = np.hstack((initialMask, mask_i))
            X_imp = np.where(X_na == self.na, initialMask, X_na)
            
        elif self.initial_imputation == 'random_SNP':
            initialMask = np.random.choice([0,1,2], size=(N,p))
            X_imp = np.where(X_na == self.na, initialMask, X_na)
            
        elif self.initial_imputation == 'mode':
            init_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
            X_imp = init_imp.fit_transform(X_na)

        if self.estimator == 'ditARF':
            # self.base_estimator.classifier.fit(X_imp, X_imp)
            self.base_estimator.dummy_fit(dimension, p)
        return X_imp


    
    
    def _addNA(self, Data, frac):
        """
        Add NA to complete data for the procedural methods
        """
        N = Data.shape[0]
        p = Data.shape[1]
            
        mask = np.zeros(N*p, dtype=bool)
        mask[:int(frac*N*p)] = True
        np.random.shuffle(mask)
        mask = mask.reshape(N, p)
        
        Data[mask] = self.na
        return Data
                        
        
        
        
    def _imputeWindow(self, X_na, sample_weights, X_prev=None, i_na=None):
        """
        Imputes missing values in one window of width $Delta$

        :param X_na: data with missing values
        :param X_prev: previous iteration imputation, for iterative methods only
        :param i_na: for 'mice' method only
        :param sample_weights: the sampling weights used in the ditARF. For the rest, they should be 1 for all instances
        """

        if self.estimator != 'ditARF':
            mdl = clone(self.base_estimator)
        else:
            mdl = self.base_estimator
        
        
        
        if self.method == 'iterative':
            
            if self.estimator.startswith('AE') or self.estimator.startswith('PCA'):
                ohe = OneHotEncoder(handle_unknown='ignore')
                X_na_ohe = ohe.fit_transform(X_na)
                X_prev_ohe = ohe.transform(X_prev)
                mdl.fit(X_na_ohe, X_prev_ohe)#, sample_weights)
                X_imp_ohe = mdl.predict(X_na_ohe)
                X_imp = ohe.inverse_transform(X_imp_ohe)
                X_imp[X_imp == None] = 0
            
            else:
                if self.estimator == 'ditARF':
                    # Here X_prev is a probabilistic dataset
                    # The sample weights are obtained when predicting
                    mdl.relearn_individual_trees(X_na, X_prev, sample_weights, sample=False)
                    # mdl.classifier.fit(X_na, X_prev, sample_weights)
                    preds_proba = mdl.predict_and_correct(X_na, X_na, self.na)
                    X_imp = np.array([np.argmax(np.array(v), axis=1).tolist() for v in preds_proba]).T
                    return X_imp, preds_proba
                else:
                    mdl.fit(X_na, X_prev, sample_weights)
                    X_imp = mdl.predict(X_na)
                
            X_imp = np.where(X_na==self.na, X_imp, X_na)
            return X_imp
        
        

        elif self.method == 'mice':

            X_train = X_na[~np.any(i_na == self.na, axis=1), :]
            X_test = X_na[np.any(i_na == self.na, axis=1), :]
            i_train = i_na[~np.any(i_na == self.na, axis=1), :]
            if i_train.shape[0] == i_na.shape[0]:
                return i_na
            
            mdl.fit(X_train, i_train)#, sample_weights)
            i_imp_test = mdl.predict(X_test)
            
            i_imp = i_na.copy()
            rows_with_na = np.any(i_imp == self.na, axis=1)
            rows_for_replacement = np.where(i_na[np.any(i_na == self.na, axis=1), 0] != self.na,
                                           i_na[rows_with_na, 0],
                                           i_imp_test)
            rows_for_replacement = rows_for_replacement.reshape(len(rows_for_replacement),1)
            i_imp[rows_with_na,:] = rows_for_replacement
            
            return i_imp


                
        elif self.method == 'procedural':
                  
            ### Select rows without m.v. for training data
            X_train = X_na[~np.any(X_na == self.na, axis=1), :]
            X_na_test = X_na[np.any(X_na == self.na, axis=1), :]
            X_prev_train = X_prev[~np.any(X_na == self.na, axis=1), :]
            
            X_imp = X_na.copy()
            
            ### If there is no missing
            if X_train.shape[0] == X_na.shape[0]:
                return X_imp

            ### If training data is empty: replace m.v. with mode for each column
            elif X_train.shape[0] == 0:
                print('No complete cases found, imputing with mode')
                for j in range(X_imp.shape[1]):
                    m = stats.mode(X_na[X_na[:,j] != self.na][:,j])[0][0]
                    X_imp[:,j] = np.where((X_imp[:,j] == self.na), m, X_imp[:,j])

            ### Else: corrupt with m.v. and train "autoreplicator"
            else:
                X_na_train = self._addNA(X_train.copy(), self.frac)
               
                if self.estimator.startswith('AE') or self.estimator.startswith('PCA'):
                    ohe = OneHotEncoder(handle_unknown='ignore')
                    X_na_train_ohe = ohe.fit_transform(X_na_train)
                    X_prev_train_ohe = ohe.transform(X_prev_train)
                    X_na_test_ohe = ohe.transform(X_na_test)
                    
                    mdl.fit(X_na_train_ohe, X_prev_train_ohe)  # MLP does not allow sample weights
                    X_imp_test_ohe = mdl.predict(X_na_test_ohe)
                    X_imp_test = ohe.inverse_transform(X_imp_test_ohe)
                    X_imp_test[X_imp_test == None] = 0
                    
                else:
                    sample_weights = np.array([1] * X_na_train.shape[0])  # Overwrite these
                    mdl.fit(X_na_train, X_prev_train, sample_weights)
                    X_imp_test = mdl.predict(X_na_test)

                rows_with_na = np.any(X_imp == self.na, axis=1)
                rows_for_replacement = np.where(X_na[np.any(X_na == self.na, axis=1), :] != self.na,
                                           X_na[rows_with_na, :],
                                           X_imp_test)
                
                X_imp[rows_with_na,:] = rows_for_replacement

            return X_imp
        
        
            
            
            
    def impute(self, X_na, dimension):
        """
        Impute missing values.
        
        Parameters
        ----------
        X_na : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data containing missing values.
        dimension: {array-like}
            Contains the dimension of each feature
            
        Returns
        -------
        X_imps : array of shape (n_iter, n_samples, n_features)
            Data where missing values are imputed, one per iteration.
        """
        
        N, p = X_na.shape[0], X_na.shape[1]
        
        X_imps = (np.zeros((1, N, p)) * (-1)).astype(int)
        
        if (self.method == 'iterative' or self.method == 'mice'):
            X_imp = self._initial_imputation(X_na, dimension)
        else:
            X_imp = X_na.copy()
        
        X_imps[0,:,:] = X_imp
        
        self.frac = -X_na[X_na == self.na].sum().sum() / X_na.size
        
        ts = []
        start = time.time()
        
        if self.estimator == 'AE':
            self.base_estimator = neural_network.MLPClassifier(hidden_layer_sizes=(int(p/10),), activation='logistic')
        if self.estimator == 'PCA':
            self.base_estimator = neural_network.MLPClassifier(hidden_layer_sizes=(int(p/10),), activation='identity')

                    
        
        imp_dist = 10
        last_loglik = 10000000  # Dummy value
        alpha = 0.005
        if self.estimator == 'ditARF':
            alpha = 2
        n = 0
        sample_weights = np.array([1] * N)  # These are the weights for the instances

        if self.method == 'procedural':
            self.max_iter = 1
        
        while imp_dist > alpha and n < self.max_iter:
            #print('iteration:', n)
            n += 1
            X_new = X_imp.copy() if self.method != 'iterative' else X_na.copy()
            
            
            if self.method == 'mice':
                for i in range (p):
                    if self.delta == 'all':
                        i_left, i_right = 0, p
                    else:
                        i_left = i - int(self.delta/2)
                        i_right = i_left + self.delta
                        if i_left < 0:
                            i_left = 0
                        if i_right > p:
                            i_right = p
                    
                    i_na = X_na[:,i].reshape(N,1)
                    X_i_na = np.hstack((X_imp[:,i_left:i], X_imp[:,i+1:i_right]))
                    i_imp = self._imputeWindow(X_i_na, i_na=i_na, sample_weights=sample_weights)
                    X_new[:,i] = i_imp.reshape(-1)
                    
                X_imp = X_new.copy()
                
                
                

            else:
                if self.delta == 'all':
                    n_windows = 1
                    i_left, i_right = 0, p
                else:
                    n_windows = int(p / self.delta)
                    i_left, i_right = 0, self.delta
                    
                for i in range(0, n_windows):
                    if self.estimator == 'ditARF':
                        X_new[:, i_left:i_right], preds_proba = self._imputeWindow(X_na[:, i_left:i_right],
                                                                      X_prev=X_imp[:, i_left:i_right],
                                                                      sample_weights=sample_weights)

                        sample_weights = np.ones(N)
                        for i in range(p):  # Per variable
                            sample_weights *= np.max(preds_proba[i], axis=1)

                    else:
                        X_new[:,i_left:i_right] = self._imputeWindow(X_na[:,i_left:i_right],
                                                                     X_prev=X_imp[:,i_left:i_right],
                                                                     sample_weights=sample_weights)
                    if self.delta != 'all':
                        i_left += self.delta
                        i_right += self.delta
                X_imp = X_new.copy()

            loglik = np.sum(np.log(sample_weights))
            self.loglikelihood.append(loglik)
        
            
            X_imps = np.vstack((X_imps, np.zeros((1,N,p))))
            X_imps[n,:,:] = X_imp
        
            if self.estimator == 'ditARF':
                if len(self.loglikelihood) > 2:
                    imp_dist = np.std(self.loglikelihood[-10:])
            else:
                imp_dist = 1 - accuracy_individual(X_imps[n, :, :][X_na == self.na],
                                                   X_imps[n - 1, :, :][X_na == self.na])

            t = time.time() - start
            ts.append(t)
            
        print('Number of iterations =', n)
        
        
            
        if self.return_time == True:
            return X_imps, ts
        else:
            return X_imps
        
