import model_runner
import numpy as np
import pandas as pd

MODEL = """
     A -> B; k1*A
      
     A = 50; 
     B = 0;
     k1 = 0.15
"""
SIMULATION_TIME = 30
NUM_POINTS = 5
COLUMNS = ['time', 'A', 'B']

def testConstructor():
  runner = model_runner.ModelRunner(MODEL,
      SIMULATION_TIME, NUM_POINTS)
  trues = [c in COLUMNS for c in runner.df_data.columns]
  assert(all(trues))
  assert(len(runner.df_noiseless) > 0)

def testGenerateObservations():
  runner = model_runner.ModelRunner(MODEL,
      SIMULATION_TIME, NUM_POINTS)
  runner.generateObservations(0.1)
  assert(len(set(runner.df_noisey.columns).symmetric_difference(
      runner.df_noiseless.columns)) == 0)
  

testGenerateObservations()
