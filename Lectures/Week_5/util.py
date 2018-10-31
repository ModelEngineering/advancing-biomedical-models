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

def generateBootstrapData(y_obs, y_fit):
  """
  Generates synthetic observations from residuals.
  :param np.array y_obs: observed data
  :param np.array y_fit: fitted data
  :return np.array: synthetic data from randomizing residuals
  """
  def generateData(y_obs, y_fit):
    """
    :param np.array y_obs:
    :param np.array y_fit:
    :return np.array: bootstrap data
    """
    residuals = y_obs - y_fit
    length = len(y_obs)
    residuals = residuals.reshape(length)
    samples = np.random.randint(0, length, length)
    result = y_fit + residuals[samples]
    result = result.reshape(length)
    return result


################### SIMULATION MODEL CLASS ########################
class SimulationModel(object):
  """Encapsulates fitting and evaluation of a simulation model."""

  def __init__(self, modelstr, params=lmfit.Parameters(), 
      start=0, stop=50, num_points=100):
    """
    :param str modelstr: antimony model
    :param int start: start of simulation
    :param int stop: stop of simulation
    :param lmfit.Parameters params: Specify name of constants in simulation
        and their initial values and range constraints
    :param int num_points: num_points of simulation
    """
    self.modelstr = modelstr
    self.std = std
    self.params = params
    self.start = start
    self.stop = stop
    self.num_points = num_points

  def simulate(self, std=0):
    """
    Simulates the model for the desired time periods using the Parameters.
    Optionally adds normally distributed noise.
    :param float std: float for the error of the standard deviation
    :return array: simulation results optionally with random errors
    """
    r = te.loada(modelstr)
    for name in params:
      param = self.params.get(name)
      r.setValue(name, param.value)
    result = r.simulate(self.start, self.stop, self.num_points)
    length = result.shape[0]
    num_response_vars = result.shape[1] - 1
    for idx in range(1, num_response_vars + 1):
      result[:, idx] = result[:, idx] + np.random.normal(0, std, length)
    return result

  def fitModel(self, response_obs, indices, weights={}):
      """
      Performs for a fit for a simulation model, handling multiple response variables.
      Parameters are model constants.
      :param NamedArray response_obs: observed values of responses
      :param array-int: for which residuals are evaluated
      :param dict weights: key is response variable; value is weight in residuals
      :return lmfit.minimizer.Minimizer: .Parameters are the parameters
      """
      def calcResiduals(params):
        results = self.simulate(self.modelstr, params)
        residuals = np.repeat(0, len(indices))
        for name in results.colnames[1:]:
            multiplier = 1.0
            if name in weights.keys():
                multiplier = weights[name]*results[name]
            residuals += multiplier*(response_obs[name][indices] - results[name][indices])
          return residuals
      #
      fitter = lmfit.Minimizer(calcResiduals, self.params)
      return fitter

  @staticmethod
  def aggregateParameters(parameters_collection):
    """
    Computes the average value of a list of parameters.
    :param list-Parameters parameters_collection:
    :return Parameters, pd.DataFrame: values are the average of the list, dataframe of parameter statistics
    Notes:
      1. Get exception if names are not the same in all Parameters
    """
    getValues(name):
      return [p.get(name).value for p in parameters_collection]
    #
    names = set([])
    for parameters in parameters_collection:
      names.union([p in parameters])
    #
    parameter_dict = {
        'name': [],
        'avg': [],
        'std': [],
        }
    for name in names:
      values = getValues(name)
      parameter_dict['name'] = name
      parameter_dict['avg'] = np.mean(values)
      parameter_dict['std'] = np.std(values)/np.sqrt(len(values))
    #
    df = pd.DataFrame(parameter_dict)
    df.set_index('name')
    result = lmfit.Parameters()
    for name in names:
      value = df.loc[name, 'avg']
      result.add(name, value=value)
    #
    return result, df
