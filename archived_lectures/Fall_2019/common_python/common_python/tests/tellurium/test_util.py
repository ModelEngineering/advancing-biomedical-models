from common_python.tellurium import util

import pandas as pd
import numpy as np
import unittest


class TestFunctions(unittest.TestCase):

  def testDfToSer(self):
    data = range(5)
    df = pd.DataFrame({'a': data, 'b': [2*d for d in data]})
    ser = util.dfToSer(df)
    assert(len(ser) == len(df.columns)*len(df))
  
  def testDfToSer(self):
    data = range(5)
    df = pd.DataFrame({'a': data, 'b': [2*d for d in data]})
    ser = util.dfToSer(df)
    assert(len(ser) == len(df.columns)*len(df))

  def testInterpolateTime(self):
    MAX = 10
    SER = pd.Series(range(MAX), index=range(MAX))
    self.assertEqual(util.interpolateTime(SER, 0.4), 0.4)
    self.assertEqual(util.interpolateTime(SER, -1), 0)
    self.assertEqual(util.interpolateTime(SER, MAX), MAX-1)


if __name__ == '__main__':
  unittest.main()
