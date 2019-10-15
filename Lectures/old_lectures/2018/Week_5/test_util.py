"""
Tests for simulation utility functions.
To run: python test_util.py
"""

import util

import lmfit
import numpy as np
import unittest

ka = 0.4
v0 = 10
kb = 0.32
kc = 0.4
PARAMETERS = lmfit.Parameters()
NAMES = ["ka", "v0", "kb", "kc"]
for name in NAMES:
  if name[0] == "v":
    maxval = 20
  else:
    maxval = 2
  PARAMETERS.add(name, value=eval(name), min=0, max=maxval)
PARAMETERS_COLLECTION = [PARAMETERS for _ in range(10)]
IGNORE_TEST = True

class TestFunctions(unittest.TestCase):

  def testFoldGenerator(self):
    NUM_FOLDS = 4
    generator = util.foldGenerator(10, NUM_FOLDS)
    size = len([g for g in generator])
    self.assertEqual(size, NUM_FOLDS)

  def testAggregateParameters(self):
    parameters = util.aggregateParameters(PARAMETERS_COLLECTION)
    self.assertTrue(isinstance(parameters, lmfit.Parameters))
    for name in parameters:
      self.assertTrue(np.isclose(
          PARAMETERS.get(name), parameters.get(name)
          ))

  def testMakeParametersStatistics(self):
    parameters = util.aggregateParameters(PARAMETERS_COLLECTION)
    result = util.makeParametersStatistics(PARAMETERS_COLLECTION)
    for name in result.keys():
      self.assertEqual(len(result[name]), 2)
      self.assertTrue(np.isclose(result[name][0], parameters.get(name)))

  def testPlotFit(self):
    # Smoke test
    util.plotFit(range(10), range(10), is_plot=False)

  def testGenerateBootstrapData(self):
    NUM = 1000
    STD = 1.0
    y_fit = np.array(range(NUM))
    y_obs = y_fit + np.random.normal(0, STD, NUM)
    for _ in range(10):
      y_new = util.generateBootstrapData(y_obs, y_fit)
      self.assertEqual(len(y_new), len(y_fit))
      self.assertTrue(np.std(y_new - y_fit), STD)

  def testGetParameterData(self):
    result = util.getParameterData(PARAMETERS_COLLECTION)
    for name in NAMES:
      self.assertTrue(name in result)
      self.assertTrue(np.isclose(np.std(result[name]), 0.0))

if __name__ == "__main__":
  unittest.main()
