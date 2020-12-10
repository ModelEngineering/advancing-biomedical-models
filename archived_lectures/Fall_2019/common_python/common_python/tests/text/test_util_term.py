"""Tests for Term Analyzer."""

from common_python.text import util_text
from common_python.types.extended_list import ExtendedList
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False

GROUPS =  ['a', 'b', 'b', 'c', 'a', 'a']
TERMS = ['x', 'x', 'y', 'x', 'x', 'x']
DF = pd.DataFrame({
    util_text.GROUP: GROUPS,
    util_text.TERM: TERMS,
    })
DF = DF.set_index(util_text.GROUP)


class TestFunctions(unittest.TestCase):

  def testMakeTermMatrix(self):
    df = util_text.makeTermMatrix(DF[util_text.TERM])
    self.assertEqual(df.loc['a', 'x'], 3.0)
    terms = ExtendedList(TERMS)
    terms.unique()
    self.assertTrue(helpers.isValidDataFrame(df, terms))
    groups = ExtendedList(GROUPS)
    groups.unique()
    self.assertTrue(groups.isSame(df.index))


if __name__ == '__main__':
  unittest.main()
