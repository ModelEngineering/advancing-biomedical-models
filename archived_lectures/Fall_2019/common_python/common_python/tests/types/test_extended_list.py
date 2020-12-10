"""Tests for Extended list"""

from common_python.types.extended_list import ExtendedList

import unittest

IGNORE_TEST = False
LIST = [0, 1, 2, 2, 3]


class TestPersister(unittest.TestCase):

  def setUp(self):
    self.extended_list = ExtendedList(LIST)

  def testRemoveAll(self):
    self.extended_list.removeAll(2)
    self.assertEqual(self.extended_list.count(2), 0)

  def testUnique(self):
    self.extended_list.unique()
    for ele in LIST:
      self.assertTrue(self.extended_list.count(ele) == 1)

  def testIsSubset(self):
    extended1 = ExtendedList([0, 1, 2])
    extended2 = ExtendedList([0, 1, 2, 3])
    self.assertTrue(extended1.isSubset(extended2))
    self.assertTrue(extended1.isSubset(extended1))

  def testIsSame(self):
    extended1 = ExtendedList([0, 1, 2])
    extended2 = ExtendedList([2, 1, 0])
    extended3 = ExtendedList([2, 1])
    self.assertTrue(extended1.isSame(extended2))
    self.assertFalse(extended3.isSame(extended1))
    self.assertFalse(extended1.isSame(extended3))


if __name__ == '__main__':
  unittest.main()
