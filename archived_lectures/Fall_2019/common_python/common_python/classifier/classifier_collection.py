"""Collection of similar classifiers and operations on them."""

import common_python.constants as cn
from common_python.util import util

import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CrossValidationResult = collections.namedtuple(
    "CrossValidationResult",
    "mean std collection"
    )


class ClassifierCollection(object):

  def __init__(self, clfs=None, features=None, 
      classes=None, scores=None):
    """
    :param list-Classifier clfs: methods-fit, predict, score
    :param list-object features:
    :param list-object classes:
    :param list-float scores:
    """
    self.clfs = util.setList(clfs)
    self.features = util.setList(features)
    self.classes = util.setList(classes)
    # Score for each classifier if present
    self.scores = util.setList(scores)

  def update(self, other):
    """
    Updates the values of the collection with those in other.
    :param ClassifierCollection other:
    """
    self.clfs = other.clfs
    self.features = other.features
    self.classes = other.classes
    self.scores = other.scores

  @classmethod
  def _partitionIndices(cls, container, all_indices, test_indices):
    """
    Partitions data indices into test and training instances based on
        specified test indicies.
    :param pd.DataFrame/pd.Series container:
    :param list all_indices:
    :param list test_indices:
    :return pd.DataFrame/pd.Series, pd.DataFrame/pd.Series:
    """
    train_indices = list(set(all_indices).difference(test_indices))
    if isinstance(container, pd.DataFrame):
      container_test = container.loc[test_indices, :]
      container_train = container.loc[train_indices, :]
    else:
      container_test = container.loc[test_indices]
      container_train = container.loc[train_indices]
    return container_train, container_test

  @classmethod
  def _make(cls, clf, df_X, ser_y, count, selTestIndicesFunc):
    """
    Creates a collection of fitted models using the provided
    function to select test indices.
    :param Classifier clf: untrained clf with fit, score methods
    :param pd.DataFrame df_X: columns of features, rows of instances
    :param pd.Series ser_y: state values
    :param int count: number of models to create
    :param function selTestIndicesFunc:
         Selects test holdout indices from randomly order indices
         input is pd.Series for class; output is list of indices
    :param dict kwargs: optional arguments passed to classifier
    :return ClassifierCollection:
    Notes
        1. df_X, ser_y must have the same index
    """
    def copyContainers(*containers):
      """
      Creates copies of the containers, randomizing the order or rows.
      :param list-pd.DataFrame/pd.Series containers:
      :return list-container:
      """
      new_containers = []
      indices = np.random.permutation(containers[0].index.tolist())
      for container in containers:
        container.index = indices
        new_container = container.sort_index()
        new_containers.append(new_container)
      return new_containers
    # Initializations
    clfs = []
    scores = []
    classes = ser_y.unique()
    dff_X = df_X.copy()
    serr_y = ser_y.copy()
    indices = dff_X.index.tolist()
    # Construct the fitted classifiers
    for _ in range(count):
      # Construct test set
      new_clf = copy.deepcopy(clf)
      dff_X, serr_y = copyContainers(dff_X, serr_y)
      test_indices = selTestIndicesFunc(serr_y)
      df_X_train, df_X_test = cls._partitionIndices(dff_X, indices, test_indices)
      ser_y_train, ser_y_test = cls._partitionIndices(serr_y, indices, test_indices)
      # Train the clf and evaluate the clf
      new_clf.fit(df_X_train, ser_y_train)
      clfs.append(new_clf)
      scores.append(new_clf.score(df_X_test, ser_y_test))
    return cls(clfs, df_X.columns.tolist(), classes, scores=scores)

  @classmethod
  def makeByRandomHoldout(cls, clf, df_X, ser_y, count, holdouts=1):
    """
    Creates a collection of fitted models with randomly chosen holdouts.
    :param Classifier clf: untrained clf with fit, score methods
    :param pd.DataFrame df_X: columns of features, rows of instances
    :param pd.Series ser_y: state values
    :param int count: number of models to create
    :param int holdouts: number of instances per state in test data
    :return ClassifierCollection:
    Notes
      1. df_X, ser_y must have the same index
    """
    def selTestIndices(ser):
      """
      :param pd.Series ser:
      :return list-indices: test indices
      """
      return ser.index.tolist()[0:holdouts]
    #
    return cls._make(clf, df_X, ser_y, count, selTestIndices)

  @classmethod
  def makeByRandomStateHoldout(cls, clf, df_X, ser_y, count, holdouts=1):
    """
    Creates a collection of fitted models with randomly chosen holdouts.
    :param Classifier clf: untrained clf with fit, score methods
    :param pd.DataFrame df_X: columns of features, rows of instances
    :param pd.Series ser_y: state values
    :param int count: number of models to create
    :param int holdouts: number of instances per state in test data
    :return ClassifierCollection:
    Notes
      1. df_X, ser_y must have the same index
    """
    def selTestIndices(ser):
      """
      :param pd.Series ser:
      :return list-indices: test indices
      """
      classes = ser.unique().tolist()
      classes.sort()
      test_indices = []
      for cl in classes:
        serr = ser[ser == cl]
        if len(serr) <= holdouts:
          raise ValueError("Class %s has fewer than %d holdouts" %
              (cl, holdouts))
        idx = serr.index[0:holdouts].tolist()
        test_indices.extend(idx)
      return test_indices
    #
    return cls._make(clf, df_X, ser_y, count, selTestIndices)

  def crossValidate(self):
    """
    Computes cross validation statistics for a collection
    :return float, float:
    """
    return np.mean(self.scores), np.std(self.scores)

  @classmethod
  def crossValidateByState(cls, clf, df_X, ser_y, num_clfs):
    """
    Does cross validation for a classification class
    that supports the fit and score methods.
    :param Classifier clf: Instantiated classifier
    :param pd.DataFrame df_X: feature matrix
    :param pd.DataFrame ser_y: classes
    :param int num_clfs: Number of classifiers to create
    :return CrossValidationResult:
    """
    collection = cls.makeByRandomStateHoldout(clf, df_X, ser_y,
         num_clfs)
    clf_mean, clf_std = collection.crossValidate()
    return CrossValidationResult(mean=clf_mean, std=clf_std,
        collection=collection)
