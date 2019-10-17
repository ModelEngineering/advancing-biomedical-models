"Runs a tellurium model"

import tellurium as te
import numpy as np
import pandas as pd
import lmfit   # Fitting lib
import random 
import matplotlib.pyplot as plt

# Column names
TIME = 'time'
MEAN = 'mean'
STD = 'std'

# Globals
runner = None

############ FUNCTIONS ######################
def residuals(parameters):
  """
  Computes residuals from the parameters using the global runner
  Residuals are computed based on the values of chemical species
  in the model.
  :param lmfit.Parameter parameters:
  """
  df_sim_data, _ = runner.runSimulation(parameters=parameters)
  df_residuals = runner.df_observation - df_sim_data
  ser = dfToSer(df_residuals)
  return np.array(ser.tolist())

def dfToSer(df):
  """
  Converts a dataframe to a series.
  :param pd.DataFrame df:
  :return pd.Series:
  """
  return pd.concat([df[c] for c in df.columns])


############ CLASSES ######################
class ModelRunner(object):

  def __init__(self, model_str, constants,
      simulation_time, num_points, noise_std=0.5):
    """
    :param str model_str: Antimony model
    :param list-str constants: list of constants to fit in model
    :param int simulation_time: length of simulation
    :param int num_points: number of data points
    """
    self.model_str = model_str
    self.road_runner = te.loada(self.model_str)
    self.constants = constants
    self.simulation_time = simulation_time
    self.num_points = num_points
    self.noise_std = noise_std
    self.df_observation, self.ser_time = self.generateObservations()
    self.species = self.df_observation.columns.tolist()
    self.df_noisey = None

  def runSimulation(self, parameters=None):
    """
    Runs a simulation.
    :param Parameters parameters: If None, use existing values.
    :return pd.Series, pd.DataFrame: time, concentrations
    """
    self.road_runner.reset()
    if parameters is not None:
      # Set the value of constants in the simulation
      param_dict = parameters.valuesdict()
      for constant in param_dict.keys():
        stmt = "runner.road_runner.%s = param_dict['%s']" % (
            constant, constant)
        exec(stmt)
    #
    data = self.road_runner.simulate(0,
        self.simulation_time, self.num_points)
    # Construct the data frames
    df_alldata = pd.DataFrame(data)
    columns = [c[1:-1] for c in data.colnames]  # Eliminate square brackets
    columns[0] = TIME
    df_alldata.columns = columns
    ser_time = df_alldata[TIME]
    df_data = df_alldata[df_alldata.columns[1:]]
    return df_data, ser_time

  def generateObservations(self, parameters=None, std=None):
    """
    Creates random observations by adding normally distributed
    noise.
    :param float std: if none, use constructor
    :return pd.DataFrame, ser_times: noisey data, time
    """
    df_data, ser_time = self.runSimulation(parameters=parameters)
    if std is None:
      std = self.noise_std
    df_rand = pd.DataFrame(np.random.normal(0, std,
         (len(df_data),len(df_data.columns))))
    df_rand.columns = df_data.columns
    df = df_data + df_rand
    df = df.applymap(lambda v: 0 if v < 0 else v)
    return df, ser_time
  
  # FIXME: Allow for specifying min/max for constants 
  def fit(self, count=1, method='leastsq', std=None, func=None):
    """
    Performs multiple fits.
    :param int count: Number of fits to do, each with different
                      noisey data
    :return pd.DataFrame: columns species; rows are MEAN, STD
    Assigns value to df_noisey to communicate with func
    """
    if func is None:
      global runner
      runner = self
      func = residuals
    if std is None:
      std = self.noise_std
    #
    estimates = {}
    for constant in self.constants:
        estimates[constant] = []  # Initialize to empty list
    # Do the analysis multiple times with different observations
    for _ in range(count):
      self.df_observation, self_df_time = self.generateObservations()
      parameters = lmfit.Parameters()
      for constant in self.constants:
          parameters.add(constant, value=1, min=0, max=10)
      # Create the minimizer
      fitter = lmfit.Minimizer(func, parameters)
      result = fitter.minimize (method=method)
      for constant in self.constants:
        estimates[constant].append(
           result.params.get(constant).value)
    df_estimates = pd.DataFrame(estimates)
    df_result = pd.DataFrame()
    df_result[MEAN] = df_estimates.mean(axis=0)
    df_result[STD] = df_estimates.std(axis=0)
    return df_result
        
