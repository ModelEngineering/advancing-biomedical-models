import lmfit
import model_runner
import numpy as np
import pandas as pd


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

def testConstructor():
  runner = model_runner.ModelRunner(MODEL, CONSTANTS,
      SIMULATION_TIME, NUM_POINTS)
  trues = [c in COLUMNS for c in runner.df_observation.columns]
  assert(all(trues))
  assert(len(runner.df_observation) > 0)

def testGenerateObservations():
  runner = model_runner.ModelRunner(MODEL, CONSTANTS,
      SIMULATION_TIME, NUM_POINTS)
  df, _ = runner.generateObservations(std=0.1)
  assert(len(set(df.columns).symmetric_difference(
      runner.df_observation.columns)) == 0)

def testResiduals():
  runner = model_runner.ModelRunner(MODEL, CONSTANTS,
      SIMULATION_TIME, NUM_POINTS)
  runner.df_observation, _ = runner.generateObservations(std=0.1)
  model_runner.runner = runner
  parameters = lmfit.Parameters()
  for constant in CONSTANTS:
    parameters.add(constant, value=1, min=0, max=10)
  residuals = model_runner.residuals(parameters)
  assert(len(residuals) == NUM_POINTS)

def testFit():
  runner = model_runner.ModelRunner(MODEL1, CONSTANT1S,
      SIMULATION_TIME, NUM_POINTS, noise_std=0.0)
  df = runner.fit(count=20)
  import pdb; pdb.set_trace()
  assert(len(df.columns) == 2)
  assert(len(df) == len(CONSTANTS))
  
if True:
  testConstructor()
  testGenerateObservations()
  testResiduals()
  testFit()
  print("OK")
