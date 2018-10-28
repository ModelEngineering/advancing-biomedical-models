import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def foldGenerator(num_points, num_folds):
  """
  Generates indices for test and training data.
  :param int num_points:
  :param int num_folds:
  :return array, array: training indices, test indices
  """
  indices = range(num_points)
  for remainder in range(num_folds):
    test_indices = []
    for idx in indices:
      if idx % num_folds == remainder:
          test_indices.append(idx)
    train_indices = np.array(
        list(set(indices).difference(test_indices)))
    test_indices = np.array(test_indices)
    yield train_indices, test_indices

def makeSyntheticData(modelstr, std=1.0, 
    start=0, stop=50, num_points=100):
  """
  Creates synthetic data for a simulation by adding noise
  to the response variables.
  :param str modelstr: antimony model
  :param float std: float for the error of the standard deviation
  :param int start: start of simulation
  :param int stop: stop of simulation
  :param int num_points: num_points of simulation
  :return np.namedArray: columns have 'time' followed by the species
  """
  r = te.loada(model)
  result = r.simulate(start, stop, num_points)
  
