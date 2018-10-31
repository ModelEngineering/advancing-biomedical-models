"""Tests for simulation utility functions."""

import util
import unittest
import lmfit

ka = 0.4
v0 = 10
kb = 0.32
kc = 0.4
PARAMETERS = lmfit.Parameters()
for name in ["ka", "v0", "kb", "kc"]:
  if name[0] == "v":
    maxval = 20
  else:
    maxval = 2
  PARAMETERS.add(name, value=eval(name), min=0, max=maxval)
IGNORE_TEST = True

class TestFunctions(unittest.TestCase):

  def testFoldGenerator(self):
    NUM_FOLDS = 4
    generator = util.foldGenerator(10, NUM_FOLDS)
    size = len([g for g in generator])
    self.assertEqual(size, NUM_FOLDS)

  def testPlotFit(self):
    # Smoke test
    util.plotFit(range(10), range(10), is_plot=False)

if __name__ == "__main__":
  unittest.main()
