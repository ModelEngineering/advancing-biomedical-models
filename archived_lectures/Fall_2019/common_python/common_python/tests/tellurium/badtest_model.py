# TODO: Fix or discard this module

from common_python.tellurium.model import Model
from common_python.tellurium import util

import lmfit
import numpy as np
import pandas as pd
import unittest


IS_PLOT = False

MODEL = """
     A -> B; k1*A
      
     A = 50; 
     B = 0;
     C = 0;
     k1 = 0.15
"""
CONSTANTS = ['k1']
COLUMNS = ['time', 'A', 'B']
#
MODEL = """
     A -> B; k1*A
     B -> C; k2*B
      
     A = 50; 
     B = 0;
     C = 0;
     k1 = 0.15
     k2 = 0.25
"""
CONSTANTS = ['k1', 'k2']
COLUMNS = ['time', 'A', 'B', 'C']
#
MODEL1 = """
     A -> B; k1*A
      
     A = 50; 
     B = 0;
     k1 = 0.15
"""
CONSTANT1S = ['k1']
SIMULATION_TIME = 30
NUM_POINTS = 5
COLUMN1S = ['time', 'A', 'B']
IGNORE_TEST = True


class TestModel(unittest.TestCase):

  def setUp(self):
   if IGNORE_TEST:
     return
   self.model = Model(MODEL, CONSTANTS,
      SIMULATION_TIME, NUM_POINTS)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    trues = [c in COLUMNS for c in self.model.species]
    assert(all(trues))
  
  def testRunSimulation(self):
    if IGNORE_TEST:
      return
    parameters = lmfit.Parameters()
    for constant in CONSTANTS:
      parameters.add(constant, value=1, min=0, max=10)
    df, _ = self.model.runSimulation(parameters=parameters)
    self.assertTrue(util.isEqualList(df.columns, self.model.species))

  def testCalcResiduals(self):
    if IGNORE_TEST:
      return
    df, _ = self.model.runSimulation()
    result = self.model.calcResiduals(df)
    residuals = result[0]
    self.assertTrue(all([r == 0 for r in residuals]))
    self.assertEqual(result[1], 1.0)

  def testPlotResiduals(self):
    if IGNORE_TEST:
      return
    df, _ = self.model.runSimulation()
    self.model.plotResiduals(df, is_plot=IS_PLOT)

  def testPlotData(self):
    if IGNORE_TEST:
      return
    self.model.plotData(is_plot=IS_PLOT)
    self.model.plotData(is_plot=IS_PLOT, is_scatter=True)


if __name__ == '__main__':
  unittest.main()
