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
  Residuals are computed based on the last chemical species
  in the model.
  :param lmfit.Parameter parameters:
  """
  runner.road_runner.reset()  
  param_dict = parameters.valuesdict()
  for constant in runner.constants:
    stmt = "runner.road_runner.%s = param_dict['%s']" % (
        constant, constant)
    exec(stmt)
  sim_data = runner.road_runner.simulate(0,
       runner.simulation_time, runner.num_points)
  df_sim_data = pd.DataFrame(sim_data)
  del df_sim_data[df_sim_data.columns.tolist()[0]]  # Delete time
  df_sim_data.columns = runner.species
  df_residuals = runner.df_noisey - df_sim_data
  ser = df_residuals[df_residuals.columns[-1]]
  return np.array(ser.tolist())


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
    self.constants = constants
    self.simulation_time = simulation_time
    self.num_points = num_points
    self.noise_std = noise_std
    self._makeNoiselessData()


  def _makeNoiselessData(self):
    self.road_runner = te.loada(self.model_str)
    data = self.road_runner.simulate(0,
        self.simulation_time, self.num_points)
    self.df_data = pd.DataFrame(data)
    columns = [c[1:-1] for c in data.colnames]
    columns[0] = TIME
    self.df_data.columns = columns
    self.df_noiseless = self.df_data[self.df_data.columns[1:]]
    self.species = self.df_noiseless.columns.tolist()
    self.ser_times = self.df_data[TIME]

  def generateObservations(self, std=None):
    """
    Creates random observations by adding normally distributed
    noise.
    :param float std: if none, use constructor
    :return pd.DataFrame: noisey data
    """
    if std is None:
      std = self.noise_std
    df_rand = pd.DataFrame(np.random.normal(0, std,
         (len(self.df_noiseless),len(self.species))))
    df_rand.columns = self.species
    df = self.df_noiseless + df_rand
    df = df.applymap(lambda v: 0 if v < 0 else v)
    return df
  
  # FIXME: Allow for specifying min/max for constants 
  def fit(self, count=1, method='leastsq', std=None, func=None):
    """
    Performs multiple fits.
    :param int count: Number of fits to do, each with different
                      noisey data
    :return pd.DataFrame: columns species; rows are MEAN, STD
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
    for _ in range(count):
      self.df_noisey = self.generateObservations(std=std)
      parameters = lmfit.Parameters()
      for constant in self.constants:
          parameters.add(constant, value=1, min=0, max=10)
      # Create the minimizer
      fitter = lmfit.Minimizer(residuals, parameters)
      result = fitter.minimize (method=method)
      for constant in self.constants:
        estimates[constant].append(
           result.params.get(constant).value)
    df_estimates = pd.DataFrame(estimates)
    df_result = pd.DataFrame()
    df_result[MEAN] = df_estimates.mean(axis=0)
    df_result[STD] = df_estimates.std(axis=0)
    return df_result
        
