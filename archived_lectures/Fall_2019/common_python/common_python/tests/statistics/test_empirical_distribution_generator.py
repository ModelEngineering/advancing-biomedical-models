"""Tests for Statistics Utilities."""

from common_python.statistics import empirical_distribution_generator
import common_python.constants as cn
from common_python.testing import helpers

import numpy as np
import pandas as pd
import unittest


def makeData(size):
  return pd.DataFrame({
      COLA: range(size),
      COLB: range(size),
      })

IGNORE_TEST = False
IS_PLOT = False
SIZE = 10
COLA = "colA"
COLB = "colB"
DF = makeData(SIZE)
MAX_FRAC_DIFF = 0.05
NUM_SAMPLES = 5000
TOLERANCE = 0.001


class TestEmpiricalDistributionGenerator(unittest.TestCase):

  def setUp(self):
    self.cls = empirical_distribution_generator.EmpiricalDistributionGenerator
    self.empirical = self.cls(DF)

  def testSample(self):
    if IGNORE_TEST:
      return
    df = self.empirical.sample(NUM_SAMPLES, is_decorrelate=False)
    frac = 1.0/SIZE
    frac_0 = (1.0*len(df[df[COLA] == 0])) / len(df)
    self.assertLess(np.abs(frac - frac_0), MAX_FRAC_DIFF)
    self.assertTrue(helpers.isValidDataFrame(df, DF.columns))
    df = self.empirical.sample(NUM_SAMPLES, is_decorrelate=True)
    self.assertTrue(helpers.isValidDataFrame(df, DF.columns))

  def testSynthesize(self):
    if IGNORE_TEST:
      return
    size = 500
    frac = 0.2
    df = self.empirical.synthesize(size, frac)
    expected = (1 - (1-frac)**2) * size
    count = sum([1 if r[COLA] != r[COLB] else 0 
        for _, r in df.iterrows()])
    normalized_difference = abs(count - expected) / (1.0*expected)
    self.assertLess(normalized_difference, 0.4)
    df = self.empirical.synthesize(-1, 0.0)
    self.assertEqual(len(df), len(self.empirical._df))

if __name__ == '__main__':
  unittest.main()
