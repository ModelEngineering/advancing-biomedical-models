"""Tests for classifier utilities."""

from common_python.classifier.classifier_collection  \
    import ClassifierCollection
from common_python.testing import helpers
import common_python.constants as cn
from common.trinary_data import TrinaryData
from common_python.tests.classifier import helpers as test_helpers

import pandas as pd
import numpy as np
from sklearn import svm
import unittest
import warnings

IGNORE_TEST = False
SIZE = 10
values = list(range(SIZE))
values.extend(values)
DF = pd.DataFrame({
    'A': values,
    'B': np.repeat(1, 2*SIZE),
    })
SER = pd.Series(values)
DATA = TrinaryData()
EMPTY_LIST = []


class TestClassifierCollection(unittest.TestCase):

  def setUp(self):
    self.clf = svm.LinearSVC()
    self.lin_clf = svm.LinearSVC()

  def testConstructor(self):
    if IGNORE_TEST:
      return
    collection = ClassifierCollection(EMPTY_LIST,
        EMPTY_LIST, EMPTY_LIST)
    for item in ["clfs", "features", "classes"]:
      statement = "isinstance(%s, list)" % item
      self.assertTrue(statement)

  def testMakeByRandomStateHoldout2(self):
    df_X, ser_y = test_helpers.getData()
    holdouts = 1
    result = ClassifierCollection.crossValidateByState(
        self.clf, df_X, ser_y, num_clfs=100, holdouts=holdouts)
    self.assertEqual(len(df_X.columns), 
        len(result.collection.features))
    self.assertGreater(result.mean, 0.95)

  def makeTester(self, method):
    if IGNORE_TEST:
      return
    def test(holdouts):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
          collection = method(
              self.clf, DF, SER, 10, holdouts=holdouts)
          mean, std = collection.crossValidate()
          for value in [mean, std]:
            self.assertTrue(isinstance(value, float))
          self.assertTrue(isinstance(collection,
              ClassifierCollection))
        except ValueError:
          raise ValueError
    #
    test(1)
    with self.assertRaises(ValueError):
      test(len(DF))
      pass

  def testMakeByRandomStateHoldout2(self):
    self.makeTester(
        ClassifierCollection.makeByRandomStateHoldout)

  def testMakeByRandomHoldout(self):
    self.makeTester(
        ClassifierCollection.makeByRandomHoldout)


if __name__ == '__main__':
  unittest.main()
