"""Tests for Statistics Utilities."""

from common_python.statistics import util_statistics

import numpy as np
import pandas as pd
import unittest

IGNORE_TEST = False
SIZE = 10
DF = pd.DataFrame({
    'nz-1': range(SIZE),
    'z-1': [1 for _ in range(SIZE)],
    'z-2': [2 for _ in range(SIZE)],
    'nz-2': range(SIZE),
})
DF = DF.T
NZ_INDICES = [i for i in DF.index if i[0:2] == 'nz']
Z_INDICES = [i for i in DF.index if i[0] == 'z']


class TestFunctions(unittest.TestCase):

  def testFilterZeroVarianceRows(self):
    df = util_statistics.filterZeroVarianceRows(DF)
    difference = set(NZ_INDICES).symmetric_difference(df.index)
    self.assertEqual(len(difference), 0)

  def testCalcLogSL(self):
    df = util_statistics.calcLogSL(DF)
    for index in Z_INDICES:
      trues = [np.isnan(v) for v in df.loc[index, :]]
    self.assertTrue(all(trues))
    #
    columns = df.columns
    for index in NZ_INDICES:
      for nn in range(2, len(df.loc[index, :])):
        self.assertLess(df.loc[index, columns[nn-1]],
            df.loc[index, columns[nn]])

    def testDecorelate(self):
      if IGNORE_TEST:
        return
      df_orig = pd.concat([DF for _ in range(500)], axis=1)
      df_orig.columns = [
          "%d" % d for d in range(len(df_orig.columns))]
      df = util_statistics.decorrelate(df_orig)
      self.assertTrue(helpers.isValidDataFrame(df, df_orig.columns))
      

if __name__ == '__main__':
  unittest.main()
