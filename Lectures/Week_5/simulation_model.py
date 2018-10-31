"""Instantiates simulations."""

import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import lmfit

import inspect

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


class OverrideError(Exception):
  pass


class SimulationModel(object):
  """Encapsulates fitting and evaluation of a simulation model."""

  def __init__(self, model_cls, indices=None):
    """
    :param ModelDefinitionClass model_cls:
    :param int start: start of simulation
    :param int stop: stop of simulation
    :param lmfit.Parameters params: Specify name of constants in simulation
        and their initial values and range constraints
    :param int num_points: num_points of simulation
    """
    self.model_cls = model_cls
    if indices is None:
      indices = range(num_points)
    self.indices = indices

  def fit(self, response_obs):
      """
      Performs for a fit for a simulation model, handling multiple 
      response variables.
      Parameters are model constants.
      :param NamedArray response_obs: observed values of responses
      :param array-int: for which residuals are evaluated
          if None, use all.
      :param dict weights: key is response variable; 
          value is weight in residuals
      :return lmfit.minimizer.Minimizer: .Parameters is parameters
      """
      if indices is None:
        self.indices = range(self.num_points)
      else:
        self.indices = indices
      #
      model = lmfit.Model(self.model_cls.calculate)
      data = np.repeat(0, self.num_points)
      fitter = model.fit(data, self.params)
      return fitter

  def simulate(self, params, 
      start=0, stop=50, num_points=100, params, std=0):
    """
    Simulates the model for the desired time periods using the Parameters.
    Optionally adds an error term.
    :param str modelstr: model
    :param Parameters params:
    :param int start:
    :param int stop:
    :param int num_points:
    :param float std: float for the error of the standard deviation
    :return array: simulation results optionally with random errors
    """
    r = te.loada(modelstr)
    for name in params:
      param = self.params.get(name)
      r.setValue(name, param.value)
    result = r.simulate(start, stop, num_points)
    length = result.shape[0]
    num_response_vars = result.shape[1] - 1
    for idx in range(1, num_response_vars + 1):
      result[:, idx] = result[:, idx] + np.random.normal(0, std, length)
    return result

  @staticmethod
  def _getSpeciesKey(species):
    return "{%s]" % species

  @classmethod
  def calculateResiduals(cls, model_cls, data, params):
    """
    Calculates residuals for the model.
    :param ModelDefinitionClass model_cls:
    :param NamedArray data:
    :param lmfit.Parameters params:
    :return array-float: 
    """
    results = SimulationModel.simulate(model_cls.MODELSTR, params,
        model_cls.START, model_cls.STOP, model_cls.num_points)
    residuals = np.repeat(0, len(model.indices))
    for name in results.colnames[1:]:
      multiplier = 1.0
      key = cls._getSpeciesKey(name)
      if key in model_cls.WEIGHTS.keys():
          multiplier = model_cls.WEIGHTS[key]*results[key]
      residuals += multiplier*(data[name][indices]  \
          - results[name][indices])
    import pdb; pdb.set_trace()
    return residuals

  @staticmethod
  def aggregateParameters(parameters_collection):
    """
    Computes the average value of a list of parameters.
    :param list-Parameters parameters_collection:
    :return Parameters, pd.DataFrame: values are the average of the list, dataframe of parameter statistics
    Notes:
      1. Get exception if names are not the same in all Parameters
    """
    def getValues(name):
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


#################################
class SimulationDefinition(object):
  """Provides function for lmfit"""
  MODELSTR = None
  START = 0
  END = 50
  NUM_POINTS = 100
  
  @staticmethod
  def calculate():
    raise OverrideError("Must override")


class SimulationDefinition1(SimulationDefinition):

  MODELSTR =model = """
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
  WEIGHTS = {"B": 1.0}  # Only consider the concentration of B

  @staticmethod
  def calculate(data, ka, v0, kb, kc):
    """
    Interface to lmfit.
    :param array data: observed values
    """
    params = lmfit.Parameters()
    for name in inspect.getargs():
      value = eval(name)
      params.add(name, value=value, min=0)
    return SimulationModel.calculateResiduals(
        SimulationDefinition1, data, params)
