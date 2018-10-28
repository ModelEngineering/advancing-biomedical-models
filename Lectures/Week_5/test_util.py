"""Tests for simulation utility functions."""

import util
import unittest

class TestFunctions(unittest.TestCase):

  def testFoldGenerator(self):
    NUM_FOLDS = 4
    generator = util.foldGenerator(10, NUM_FOLDS)
    size = len([g for g in generator])
    self.assertEqual(size, NUM_FOLDS)

if __name__ == "__main__":
  unittest.main()
