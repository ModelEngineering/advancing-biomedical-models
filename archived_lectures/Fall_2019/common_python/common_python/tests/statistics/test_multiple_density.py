"""Tests for Statistics Utilities."""

from common_python.statistics import multiple_density
import common_python.constants as cn
from common_python.testing import helpers

import numpy as np
import pandas as pd
import unittest


IGNORE_TEST = False
IS_PLOT = True
SIZE = 10
MAX = 5
COLA = "colA"
COLB = "colB"


def makeData(maxint, numrows, ncols=2, offset=1,
    is_random=False):
  """
  Creates a data frame for test data.
  :param int maxint: maximum value in a column
  :param int ncols: number of columns
  :param int offset: offset to the maximum
      for the second column
  :param int numrows: number of rows
  """
  data = {}
  for colidx in range(ncols):
    key = "COL%s" % cn.UPPER_CASE[colidx]
    if is_random:
      values = np.random.randint(offset, maxint, numrows)
    else:
      if colidx > 0:
        values = np.repeat(range(1, maxint+1), numrows)
      else:
        values = np.repeat(range(offset, maxint+offset), numrows)
    data[key] = values
  return pd.DataFrame(data)

DF = makeData(MAX, SIZE)


class TestMultipleDensity(unittest.TestCase):

  def setUp(self):
    self.initialize()

  def initialize(self):
    self.cls = multiple_density.MultipleDensity
    self.multiple = self.cls(DF, range(1, MAX+1))
    
  def testConstructor(self):
    if IGNORE_TEST:
      return
    self.assertEqual(len(self.multiple.df), MAX)

  def testCalcSortIndex(self):
    if IGNORE_TEST:
      return
    self.initialize()
    ser = self.multiple.calcSortIndex()
    self.assertEqual(len(ser.unique()), 1)
    #
    offset = 10
    df = makeData(MAX, SIZE, offset=offset)
    multiple = self.cls(df, range(1, MAX+offset))
    ser = multiple.calcSortIndex()
    self.assertEqual(len(ser.unique()), 2)

  def testPlotMarginals(self):
    if IGNORE_TEST:
      return
    offset = 10
    ncols = 12
    df = makeData(MAX, SIZE, ncols=ncols, is_random=True)
    multiple = self.cls(df, range(1, MAX+offset))
    columns = multiple.plotMarginals(is_plot=IS_PLOT)
    self.assertEqual(set(columns), set(df.columns))

  def testPlotMarginalComparisons(self):
    if IGNORE_TEST:
      return
    offset = 1
    ncols = 12
    df = makeData(MAX, SIZE, ncols=ncols, is_random=True)
    multiple = self.cls(df, range(1, MAX+offset))
    multiple.plotMarginalComparisons(multiple, is_plot=IS_PLOT)
    

if __name__ == '__main__':
  unittest.main()
