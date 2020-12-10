"""Tests for Statistics Utilities."""

from common_python.statistics import density
import common_python.constants as cn
from common_python.testing import helpers

import numpy as np
import pandas as pd
import unittest


IGNORE_TEST = False
IS_PLOT = False
SIZE = 10
MAX_VALUE = 5
COLA = "colA"
values = np.random.randint(1, MAX_VALUE, size=20)
SER = pd.Series(values)


class TestDensity(unittest.TestCase):

  def setUp(self):
    self.cls = density.Density
    self.density = self.cls(SER)
    
  def testConstructor(self):
    if IGNORE_TEST:
      return
    expected = range(1, MAX_VALUE+1)
    self.assertTrue(set(self.density.variates).issubset(expected))
    self.assertGreater(len(self.density.ser_density), 0)
    not_valids = [v for v in self.density.ser_density
        if (v < 0.0) or (v > 1.0)]
    self.assertEqual(len(not_valids), 0)

  def testMakeDensity(self):
    if IGNORE_TEST:
      return
    variates = range(1, MAX_VALUE)
    ser = self.cls._makeDensity(SER, variates)
    self.assertTrue(all([v > 0 for v in ser]))
    #
    variates = range(0, MAX_VALUE)
    ser = self.cls._makeDensity(SER, variates)
    self.assertTrue(all(
        [v > 0 for i, v in ser.iteritems() if i > 0]))
    self.assertEqual(ser[0], 0)

  def testPlot(self):
    if IGNORE_TEST:
      return
    self.density.plot(is_plot=IS_PLOT)
    

if __name__ == '__main__':
  unittest.main()
