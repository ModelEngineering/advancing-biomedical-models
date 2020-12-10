"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier import classifier_collection
from common_python.classifier import classifier_ensemble
from common_python.classifier.classifier_ensemble  \
    import ClassifierEnsemble, ClassifierDescriptorSVM
from common_python.classifier.classifier_collection  \
    import ClassifierCollection
from common_python.testing import helpers
import common_python.constants as cn
from common_python.tests.classifier import helpers as test_helpers

import collections
import os
import pandas as pd
import random
from sklearn import svm
import numpy as np
import unittest

IGNORE_TEST = False
IS_PLOT = False
SIZE = 10
ITERATIONS = 3
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': values,
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)
TEST_FILE = "test_classifier_ensemble.pcl"


######## Helper Classes ######
# TODO: Create tests that use RandomClassifier or delete this
class RandomClassifier(object):

  def __init__(self):
    self.class_probs = None
    self.coef_ = []
    if os.path.exists(TEST_FILE):
      os.remove(TEST_FILE)

  def fit(self, df_X, ser_y):
    """
    Fits by recording probability of each class.
    :param object _:
    :param pd.Series ser_y:
    """
    class_counts = dict(collections.Counter(ser_y.values.tolist()))
    self.class_probs = {k: v / len(ser_y) 
         for k,v in class_counts.items()}
    self.coef_ = np.repeat(0.1, len(df_X.columns))

  def predict(self, df_X):
    """
    :param pd.DataFrame df_X:
    :return pd.Series:
    """
    randoms = np.random.randint(0, len(self.class_probs.keys()),
        len(df_X))
    values = [list(self.class_probs.keys())[n] for n in randoms]
    ser = pd.Series(values, index=df_X.index)
    return ser

  def score(self, df_X, ser_y):
    ser = self.predict(df_X)
    value = np.mean(ser == ser_y)
    return value


######## Test Classes ######
class TestClassifierEnsemble(unittest.TestCase):

  def _init(self):
    self.df_X, self.ser_y = test_helpers.getData()
    holdouts = 1
    self.svm_ensemble = ClassifierEnsemble(ClassifierDescriptorSVM(),
        size=SIZE, holdouts=holdouts)
  
  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.svm_ensemble.clfs), 0)
    self.assertEqual(len(self.svm_ensemble.features), 0)
    self.assertEqual(len(self.svm_ensemble.classes), 0)
    self.assertTrue(isinstance(self.svm_ensemble.clf_desc.clf,
        svm.LinearSVC))

  # TODO: Test training with featuers
  def testFit(self):
    if IGNORE_TEST:
      return
    holdouts = 1
    #
    def test(filter_high_rank):
      if filter_high_rank is None:
        filter_high_rank = len(self.df_X.columns)
      svm_ensemble = ClassifierEnsemble(ClassifierDescriptorSVM(),
        size=SIZE, holdouts=holdouts,
        filter_high_rank=filter_high_rank)
      svm_ensemble.fit(self.df_X, self.ser_y)
      for items in [svm_ensemble.clfs, svm_ensemble.scores]:
        self.assertEqual(len(items), SIZE)
      self.assertEqual(len(svm_ensemble.features), filter_high_rank)
    #
    test(None)
    test(10)

  def testOrderFeatures(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    clf = self.svm_ensemble.clfs[0]
    #
    def test(class_selection):
       result = self.svm_ensemble._orderFeatures(clf,
           class_selection=class_selection)
       self.assertEqual(len(result), len(self.svm_ensemble.features))
       trues = [ v in range(1, len(result)+1) for v in result]
       self.assertTrue(all(trues))
    #
    test(None)
    test(1)

  def testMakeRankDF(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    df = self.svm_ensemble.makeRankDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))

  def testMakeImportanceDF(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    #
    def test(class_selection):
      df = self.svm_ensemble.makeImportanceDF(
          class_selection=class_selection)
      self.assertTrue(helpers.isValidDataFrame(df,
          [cn.MEAN, cn.STD, cn.STERR]))
      trues = [df.loc[df.index[i], cn.MEAN]
           >= df.loc[df.index[i+1], cn.MEAN] 
          for i in range(len(df.index)-1)]
    #
    test(None)
    test(1)

  def testPlotImportance(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    # Smoke tests
    _ = self.svm_ensemble.plotImportance(top=40, title="SVM",
        is_plot=IS_PLOT, ylabel="XXX")
    _ = self.svm_ensemble.plotImportance(top=40, title="SVM-class 2", 
        class_selection=2, is_plot=IS_PLOT)

  def testPlotRank(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
   # Smoke tests
    _ = self.svm_ensemble.plotRank(top=40, title="SVM", is_plot=IS_PLOT)

  def testCrossValidate(self):
    # Tests the cross validation of ClassifierEnsemble
    if IGNORE_TEST:
      return
    cvrs = []
    num_clfs = 10 # number of classifiers created randomly
    for fltr in [6, 1515]:
      clf = classifier_ensemble.ClassifierEnsemble(
          classifier_ensemble.ClassifierDescriptorSVM(),
          filter_high_rank=fltr, size=10)
      cvrs.append(
          classifier_collection.ClassifierCollection.crossValidateByState(
          clf, self.df_X, self.ser_y, num_clfs))
    self.assertGreater(cvrs[1], cvrs[0])

  # TODO: Use or delete
  def testRandomClassifier(self):
    return
    clf = RandomClassifier()
    clf.fit(DF, SER)
    self.assertEqual(clf.class_probs[0], 1/SIZE)
    ser_predict = clf.predict(DF)
    trues = [c in range(SIZE) for c in ser_predict.values]
    self.assertTrue(all(trues))

  # TODO: Use or delete
  def testPredict(self):
    return
    self._init()
    ser = self.random_classifier_ensemble.predict(DF.loc[0,:])
    mean = ser.mean(axis=1).values[0]
    expected = 1/SIZE
    self.assertLess(abs(mean - expected), 0.1)

  # TODO: Use or delete
  def testScore(self):
    return
    self._init()
    score = self.random_classifier_ensemble.score(DF, SER)
    expected = 1/SIZE
    self.assertLess(abs(score- expected), 0.1)

  def testSerializeAndDeserialize(self):
    if IGNORE_TEST:
      return
    self.svm_ensemble.fit(self.df_X, self.ser_y)
    self.svm_ensemble.serialize(TEST_FILE)
    svm_ensemble = ClassifierEnsemble.deserialize(TEST_FILE)
    diff = set(self.svm_ensemble.features).symmetric_difference(
        svm_ensemble.features)
    self.assertEqual(len(diff), 0)


if __name__ == '__main__':
  unittest.main()
