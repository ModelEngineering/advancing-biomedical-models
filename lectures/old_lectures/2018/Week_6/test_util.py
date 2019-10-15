"""
Tests for simulation utility functions.
"""

import util

import numpy as np
import tesbml
import unittest

FILENUM = 64


class TestFunctions(unittest.TestCase):

  def testMakeURL(self):
    url = util.makeURL(FILENUM)
    self.assertTrue(str(FILENUM) in url)

  def testMakeReactionStrings(self):
    reaction_stg = util.makeReactionStrings()
    import pdb; pdb.set_trace()

  def testMakeModel(self):
    model = util.makeModel(FILENUM)
    self.assertTrue(isinstance(model, tesbml.libsbml.Model))
    


if __name__ == "__main__":
  unittest.main()
