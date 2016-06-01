#Implements marginalized denoising with linear regression estimator
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn import linear_model, cross_validation, grid_search

import sys, random
import msda

class MD(BaseEstimator):
  def __init__(self, prob_corruption = 0.3, clf = linear_model.LinearRegression(), clf_params = None):
    self.clf = clf
    self.prob_corruption = prob_corruption
    self.mapping = None
    if clf_params is not None:
      self.clf.set_params(clf_params)

  def fit(self, X, y): #check_estimator wants var names X, y
    md_mapping = msda.compute_reconstruction_mapping(X, self.prob_corruption)
    self.mapping = md_mapping

    md_train_data = np.dot(X, self.mapping)
    self.clf.fit(md_train_data, y)

  def predict(self,X):
    md_test_data = np.dot(X, self.mapping)
    test_predictions = self.clf.predict(md_test_data)
    return test_predictions

class mSDA(BaseEstimator):
  def __init__(self, prob_corruption = 0.3, num_layers = 5, use_nonlinearity = True, clf = linear_model.LinearRegression(), clf_params = None):
    self.clf = clf
    self.prob_corruption = prob_corruption
    self.num_layers = num_layers
    self.use_nonlinearity = use_nonlinearity
    self.mappings = None

    if clf_params is not None:
      self.clf.set_params(clf_params)

  def fit(self, X, y): #check_estimator wants var names X, y
    mappings, representations = msda.mSDA(X, self.prob_corruption, self.num_layers, self.use_nonlinearity)
    self.mappings = mappings
    msda_train_data = representations[-1]
    self.clf.fit(msda_train_data, y)

  def predict(self, X):
    maps, test_reps = msda.mSDA(X, self.prob_corruption, self.num_layers, self.use_nonlinearity, self.mappings)
    msda_test_data = test_reps[-1]
    test_predictions = self.clf.predict(msda_test_data)
    return test_predictions

class tanh(BaseEstimator):
  def __init__(self, num_layers = 5, clf = linear_model.LinearRegression(), clf_params = None):
    self.clf = clf
    self.num_layers = num_layers
    self.mappings = None

    if clf_params is not None:
      self.clf.set_params(clf_params)

  def fit(self, X, y): #check_estimator wants var names X, y
    for i in range(self.num_layers):
      X = np.tanh(X)
    self.clf.fit(X, y)

  def predict(self, X):
    for i in range(self.num_layers):
      X = np.tanh(X)
    test_predictions = self.clf.predict(X)
    return test_predictions

