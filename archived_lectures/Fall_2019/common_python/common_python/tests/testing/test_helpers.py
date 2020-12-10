"""Tests for Helpers"""

from common_python.testing import helpers as he

import numpy as np
import pandas as pd
import os
import unittest

IGNORE_TEST = False
he.DEBUG = False  # Don't use pdb when invalid dataframe


class TestFunctions(unittest.TestCase):

  def testisValidDataFrame(self):
    length = 10
    #
    data = {'a': range(length), 'b': range(length)}
    columns = list(data.keys())
    df = pd.DataFrame(data)
    self.assertTrue(he.isValidDataFrame(df, list(data.keys()),
        key=['a', 'b']))
    #
    data = {'a': range(length), 'b': range(length)}
    df = pd.DataFrame(data)
    self.assertTrue(he.isValidDataFrame(df, 
        expected_columns=columns))
    #
    df2 = pd.DataFrame(df['a'])
    self.assertFalse(he.isValidDataFrame(df2, 
        expected_columns=columns))
    #
    data.update({'c': np.repeat(np.nan, length).tolist()})
    columns = list(data.keys())


class TestMockFileDescriptor(unittest.TestCase):
  
  def setUp(self):
    self.mocker = he.MockFileDescriptor(__file__, "r")

  def tearDown(self):
    self.mocker.close()

  def testRead(self):
    result = self.mocker.read()
    self.assertTrue(isinstance(result, str))

  def testReadLines(self):
    result = self.mocker.readlines()
    self.assertTrue(isinstance(''.join(result), str))

  def testWrite(self):
    self.mocker.write()

  def testWriteLines(self):
    self.mocker.writelines()



    

if __name__ == '__main__':
    unittest.main()

