"""Tests for DataFrame Utilities."""

from common_python.dataframe import util_dataframe
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
SIZE = 10
SMALL = "small"
LARGE = "large"
DF = pd.DataFrame({
    SMALL: np.repeat(0.1, SIZE), 
    LARGE: np.repeat(1, SIZE), 
    })
DF = DF.transpose()


class TestFunctions(unittest.TestCase):

  def testDropSmallRows(self):
    df = util_dataframe.pruneSmallRows(DF, min_abs=1)
    self.assertEqual(len(df), 1)
    self.assertEqual(df.index[0], LARGE)
    self.assertEqual(len(df.columns), SIZE)


if __name__ == '__main__':
  unittest.main()
