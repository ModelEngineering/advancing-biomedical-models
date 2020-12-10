"""Tests for OrdinalCollection."""

from common_python.text.ordinal_collection import OrdinalCollection
from common_python.testing import helpers

import os
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False

OC1 =  ['c', 'b', 'a']
OC2 =  ['c', 'b', 'a', 'd']
OC3 =  ['x', 'y', 'z']
OC4 =  ['d', 'c', 'b', 'a']
OC5 =  ['c', 'b', 'a', 'd', 'e']


class TestOrdinalCollection(unittest.TestCase):

  def setUp(self):
    self.collection = OrdinalCollection(OC1)

  def testConstructor1(self):
    if IGNORE_TEST:
      return
    self.assertEqual(OC1, self.collection.ordinals)

  def testMakeWithOrderings(self):
    if IGNORE_TEST:
      return
    orderings = [ [2, 3, 1], [30, 20, 10 ]]
    cc1 = list(OC1)
    collection = OrdinalCollection.makeWithOrderings(OC1, orderings)
    cc1.sort()
    self.assertEqual(cc1, collection.ordinals)
    #
    orderings = [ [-30, -20, -10 ]]
    collection = OrdinalCollection.makeWithOrderings(OC1, orderings,
        is_abs=False)
    self.assertEqual(OC1, collection.ordinals)
    #
    orderings = [ [-30, -20, -10 ]]
    collection = OrdinalCollection.makeWithOrderings(OC1, orderings,
        is_abs=True)
    cc1 = list(OC1)
    cc1.sort()
    self.assertEqual(cc1, collection.ordinals)

  def testCompareOverlap(self):
    if IGNORE_TEST:
      return
    #
    other = OrdinalCollection(OC4)
    result = self.collection.compareOverlap([other], topN=3)
    self.assertEqual(result, 1.0)
    #
    other = OrdinalCollection(OC2)
    result = self.collection.compareOverlap([other], topN=3)
    self.assertEqual(result, 0.5)
    #
    result = self.collection.compareOverlap([other])
    expected = (1.0*len(OC1))/len(OC2)
    self.assertEqual(result, expected)

  def testMakeOrderMatrix(self):
    if IGNORE_TEST:
      return
    df = self.collection.makeOrderMatrix()
    num_ones = df.sum().sum()
    expected = self.calcCombs(self.collection)
    self.assertEqual(num_ones, expected)

  def calcCombs(self, collection):
    count = len(collection.ordinals)
    return (count-1)*count*0.5

  def testCompareOrder(self):
    if IGNORE_TEST:
      return
    result = self.collection.compareOrder([self.collection])
    self.assertEqual(result, 1.0)
    #
    collection = OrdinalCollection(OC3)
    result = self.collection.compareOrder([collection])
    self.assertEqual(result, 0)
    #
    collection = OrdinalCollection(OC5)
    result = self.collection.compareOrder([collection])
    expected = (1.0*self.calcCombs(self.collection))/(
        self.calcCombs(collection))
    self.assertEqual(result, expected)
    

if __name__ == '__main__':
  unittest.main()
