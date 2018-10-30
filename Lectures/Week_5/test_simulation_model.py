"""Tests for simulation utility functions."""

from simulation_model import SimulationModel

import lmfit
import numpy as np
import pandas as pd
import unittest

MODELSTR = """
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
"""
PARAMS = lmfit.Parameters()
PARAMS.add('v0', value=1, min=0)
PARAMS.add('ka', value=0.1, min=0)
PARAMS.add('kb', value=0.1, min=0)
PARAMS.add('kc', value=0.1, min=0)
NUM_POINTS = 150
STD = 0.5

class TestSimulationModel(unittest.TestCase):

  def setUp(self):
    self.model = SimulationModel(MODELSTR, num_points=NUM_POINTS,
        weights={"[B]": 1.0},
        params=PARAMS)

  def testConstructor(self):
    self.assertEqual(self.model.num_points, NUM_POINTS)

  def testSimulate(self):
    result1 = self.model.simulate(std=STD)
    self.assertEqual(result1.shape, (NUM_POINTS, 4))
    result2 = self.model.simulate(std=STD)
    for col in result1.colnames:
      if col != 'time':
         values = result1[col] - result2[col]
         std = np.std(values)
         self.assertLess(abs(STD-std), 0.4)

  def testFit(self):
    response_obs = self.model.simulate(std=STD)
    result = self.model.fit(response_obs)
    import pdb; pdb.set_trace()
   
    

if __name__ == "__main__":
  unittest.main()
