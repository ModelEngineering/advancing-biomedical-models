import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import lmfit


################### FUNCTIONS ########################

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

def plotFit(data, fit, is_plot=True):
  """
  Plots observed vs. fit
  :param array.float data:
  :param array.float fit:
  """
  plt.scatter(data, fit)
  plt.xlabel("Observed")
  plt.ylabel("Fitted")
  if is_plot:
    plt.show()

def makeKwargs(params):
  """
  Creates kwargs from Parameters to make function call.
  :param lmfit.Parameters
  :return dict:
  """
  kwargs = {}
  for name in params:
    param = params.get(name)
    kwargs[name] = param.value
  return kwargs
