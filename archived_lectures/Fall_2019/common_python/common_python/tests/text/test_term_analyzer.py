"""Tests for Term Analyzer."""

from common_python.text import term_analyzer
from common_python import constants as cn
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False

OCCURS = {
    'a': 2,
    'bb': 2,
    'c': 2,
    'aa': 1,
    }
LIST = []
for term, count in OCCURS.items():
  LIST.extend(list(np.repeat(term, count)))
SER = pd.Series(LIST)


class TestTermAnalyzer(unittest.TestCase):

  def setUp(self):
    self.analyzer = term_analyzer.TermAnalyzer(noise_terms=None)

  def testConstructor(self):
    trues = [x == y for x, y in 
        zip(self.analyzer._noise_terms, term_analyzer.NOISE_TERMS)]
    self.assertTrue(all(trues))

  def testMakeSingleGroupDF(self):
    self.analyzer.makeSingleGroupDF(SER)
    self.assertTrue(helpers.isValidDataFrame(self.analyzer.df_term,
        [cn.COUNT, cn.FRAC]))
    for item, row in self.analyzer.df_term.iterrows():
      self.assertEqual(row[cn.COUNT], OCCURS[item])
    DELETE_TERM = 'a'
    self.analyzer = term_analyzer.TermAnalyzer(
        noise_terms=[DELETE_TERM])
    self.analyzer.makeSingleGroupDF(SER)
    self.assertFalse(
        DELETE_TERM in self.analyzer.df_term.index.tolist())

if __name__ == '__main__':
  unittest.main()
