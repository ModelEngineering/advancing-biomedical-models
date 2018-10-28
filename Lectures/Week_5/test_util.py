"""Tests for simulation utility functions."""

import util
import unittest

class TestFunctions(unittest.TestCase):

  def testFoldGenerator(self):
    NUM_FOLDS = 4
    generator = util.foldGenerator(10, NUM_FOLDS)
    size = len([g for g in generator])
    self.assertEqual(size, NUM_FOLDS)

  def testMakeSyntheticData(self):
    modelstr = '''
    model test
        species A, B, C;

        J0: -> A; v0
        A -> B; ka*A;
        B -> C; kb*B;
        J1: C ->; C*kc
        ka = 0.4;
        v0 = 10
        kb = 0.8*ka
        kc = ka

    end
    '''
    result = util.makeSyntheticData(modelstr)
    self.assertEqual(result.shape, (100, 4))

if __name__ == "__main__":
  unittest.main()
