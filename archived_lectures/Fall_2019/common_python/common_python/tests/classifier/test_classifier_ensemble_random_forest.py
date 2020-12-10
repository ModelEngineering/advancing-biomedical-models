"""Tests for classifier_ensemble.ClassifierEnsemble."""

from common_python.classifier.classifier_ensemble  \
    import ClassifierEnsemble, ClassifierDescriptor
from common_python.classifier.classifier_ensemble_random_forest  \
    import ClassifierEnsembleRandomForest
from common_python.testing import helpers
from common_python.tests.classifier import helpers as test_helpers
import common_python.constants as cn

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import unittest
import warnings

IGNORE_TEST = False
IS_PLOT = False
SIZE = 10
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': values,
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)


class TestClassifierEnsembleRandomForest(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.df_X, self.ser_y = test_helpers.getData()
    self.ensemble = ClassifierEnsembleRandomForest()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertTrue(isinstance(self.ensemble.clf_desc.clf,
        RandomForestClassifier))

  def testFit(self):
    if IGNORE_TEST:
      return
    self.ensemble.fit(self.df_X, self.ser_y)
    self.assertEqual(self.ensemble.clfs[0].n_features_,
        len(self.df_X.columns))
    #
    TOP = 10
    ensemble = ClassifierEnsembleRandomForest(filter_high_rank=TOP)
    ensemble.fit(self.df_X, self.ser_y)
    self.assertEqual(len(ensemble.features), TOP)

  def testPredict(self):
    if IGNORE_TEST:
      return
    self.ensemble.fit(self.df_X, self.ser_y)
    df_pred = self.ensemble.predict(self.df_X)
    ser = df_pred.T.sum()
    trues = [1 == v for v in ser.values]
    self.assertTrue(all(trues))

  def testMakeRankDF(self):
    if IGNORE_TEST:
      return
    self.ensemble.fit(self.df_X, self.ser_y)
    df = self.ensemble.makeRankDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))
    self.assertGreater(df.loc[df.index[0], cn.MEAN], 0)

  def testMakeImportanceDF(self):
    if IGNORE_TEST:
      return
    self.ensemble.fit(self.df_X, self.ser_y)
    df = self.ensemble.makeImportanceDF()
    self.assertTrue(helpers.isValidDataFrame(df,
        [cn.MEAN, cn.STD, cn.STERR]))
    self.assertGreater(df.loc[df.index[0], cn.MEAN], 0)

  def testPlotRank(self):
    if IGNORE_TEST:
      return
    self.ensemble.fit(self.df_X, self.ser_y)
   # Smoke tests
    _ = self.ensemble.plotRank(top=40, title="RandomForest", is_plot=IS_PLOT)

  def testPlotImportance(self):
    if IGNORE_TEST:
      return
    ensemble = ClassifierEnsembleRandomForest(filter_high_rank=10)
    ensemble.fit(self.df_X, self.ser_y)
   # Smoke tests
    _ = ensemble.plotImportance(top=40, title="RandomForest", is_plot=IS_PLOT)


if __name__ == '__main__':
  unittest.main()
