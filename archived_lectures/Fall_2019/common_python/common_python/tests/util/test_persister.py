"""Tests for Persister."""

from common_python.util.persister import Persister
from common_python import constants as cn

import os
import unittest

IGNORE_TEST = False
TEST_PATH = os.path.join(cn.TEST_DIR, "test_persister.pcl")
OBJECT = [0, 1, 2]


class TestPersister(unittest.TestCase):

  def setUp(self):
    self.persister = Persister(TEST_PATH)

  def tearDown(self):
    if os.path.isfile(TEST_PATH):
      os.remove(TEST_PATH)

  def testConstructor(self):
    self.assertEqual(self.persister._path, TEST_PATH)

  def testIsExist(self):
    self.assertFalse(self.persister.isExist())
    self.persister.set(OBJECT)
    self.assertTrue(self.persister.isExist())

  def testGetAndSet(self):
    self.persister.set(OBJECT)
    an_object = self.persister.get()
    trues = [x == y for x, y in zip(an_object, OBJECT)]
    self.assertTrue(all(trues))

  def testRemove(self):
    self.persister.set(OBJECT)
    self.persister.remove()
    self.assertFalse(self.persister.isExist())
    
   

if __name__ == '__main__':
    unittest.main()
