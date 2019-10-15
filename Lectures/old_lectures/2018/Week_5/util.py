import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import lmfit


COL_NAME = 'name'
COL_MEAN = 'mean'
COL_STD = 'std'


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

def generateBootstrapData(y_obs, y_fit):
    """
    Generates synthetic data by resampling residuals.
    :param np.array y_obs
    :param np.array y_fit
    :return np.array: bootstrap data
    """
    residuals = y_obs - y_fit
    length = len(y_obs)
    residuals = residuals.reshape(length)
    samples = np.random.randint(0, length, length)
    result = y_fit + residuals[samples]
    result = result.reshape(length)
    return result

def getParameterData(parameters_collection):
  """
  Computes the name and values of parameters.
  :param list-Parameters parameters_collection:
  :return dict: key is name, values is list for name
  """
  def getNames(parameters):
    return [n for n in parameters]
  #
  result = {}
  names = set([])
  for parameters in parameters_collection:
    names = names.union(getNames(parameters))
  for name in names:
    result[name] = [p.get(name).value for p in parameters_collection]
  return result

def aggregateParameters(parameters_collection):
  """
  Computes the average value of a list of parameters.
  :param list-Parameters parameters_collection:
  :return Parameters:
  """
  result = lmfit.Parameters()
  for name, values in getParameterData(parameters_collection).items():
    result.add(name, value=np.mean(values))
  return result

def makeParametersStatistics(parameters_collection):
  """
  Computes the mean and std of the parameters.
  :param list-Parameters parameters_collection:
  :return dict: key is name; value is (mean, std)
  """
  result = {}
  for name, values in getParameterData(parameters_collection).items():
    result[name] = (np.mean(values), 
                    np.std(values)/np.sqrt(len(values))
                   )
  return result

def plotFit(data, fit, is_plot=True):
  """
  Plots observed vs. fit
  :param array.float data:
  :param array.float fit:
  """
  plt.scatter(data, fit)
  plt.xlabel("Observed")
  plt.ylabel("Fitted")
  line_x = [data[0], data[-1]]
  line_y = [fit[0], fit[-1]]
  plt.plot(line_x, line_y, 'r')
  if is_plot:
    plt.show()
