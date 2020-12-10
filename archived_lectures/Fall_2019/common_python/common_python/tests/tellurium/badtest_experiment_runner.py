# TODO: Fix or discard this module

from common_python.tellurium import experiment_runner

import lmfit
import numpy as np
import pandas as pd
import unittest


MODEL0 = """
     A -> B; k1*A
      
     A = 50; 
     B = 0;
     C = 0;
     k1 = 0.15
"""
CONSTANT0S = ['k1']
COLUMN0S = ['time', 'A', 'B']
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


class TestExperimentRunner(unittest.TestCase):

  def testConstructor(self):
    if IGNORE_TEST:
      return
    runner = experiment_runner.ExperimentRunner(MODEL, CONSTANTS,
        SIMULATION_TIME, NUM_POINTS)
    trues = [c in COLUMNS for c in runner.df_observation.columns]
    assert(all(trues))
    assert(len(runner.df_observation) > 0)
  
  def testGenerateObservations(self):
    if IGNORE_TEST:
      return
    runner = experiment_runner.ExperimentRunner(MODEL, CONSTANTS,
        SIMULATION_TIME, NUM_POINTS)
    df, _ = runner.makeObservations()
    assert(len(set(df.columns).symmetric_difference(
        runner.df_observation.columns)) == 0)
  
  def testFit1(self):
    if IGNORE_TEST:
      return
    for constants, model in [(CONSTANTS, MODEL), (CONSTANT1S, MODEL1)]:
      runner = experiment_runner.ExperimentRunner(model, constants,
          SIMULATION_TIME, NUM_POINTS, noise_std=0.0)
      df = runner.fit(count=20)
      self.assertEqual(len(df.columns), 2)
      self.assertEqual(len(df),  len(constants))

  def testFit2(self):
    # Consider using a subset of indices
    num_folds = 3
    num_points = num_folds*NUM_POINTS
    for fold in range(num_folds):
      indices = [i for i in range(num_points) 
          if i % num_folds == fold]
      runner = experiment_runner.ExperimentRunner(MODEL, CONSTANTS,
          SIMULATION_TIME, num_points, indices=indices,
          noise_std=0.0)
      fit_result = runner.fit(count=5, method="differential_evolution")
      df = fit_result.params
      self.assertEqual(len(df.columns), 2)
      self.assertEqual(len(df),  len(CONSTANTS))
      print("RSQ=%f" % fit_result.rsq)
      print(df)


if __name__ == '__main__':
  unittest.main()
