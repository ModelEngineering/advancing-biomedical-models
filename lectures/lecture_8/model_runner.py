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

############ FUNCTIONS ######################


############ CLASSES ######################
class ModelRunner(object):

  def __init__(self, model_str, simulation_time, num_points):
    """
    :param str model_str: Antimony model
    :param int simulation_time: length of simulation
    :param int num_points: number of data points
    """
    self.model_str = model_str
    self.simulation_time = simulation_time
    self.num_points = num_points
    self._makeData()

  def _makeData(self):
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
    self.df_noisey = pd.DataFrame()

  def generateObservations(self, std):
    """
    Creates random observations by adding normally distributed
    noise.
    :param float std:
    """
    df_rand = pd.DataFrame(np.random.normal(0, std,
         (len(self.df_noiseless),len(self.species))))
    df_rand.columns = self.species
    self.df_noisey = self.df_noiseless + df_rand
    self.df_noisey = self.df_noisey.applymap(lambda v:
        0 if v < 0 else v)
   
  def fit(self, count=1, method='leastsq', std=0.5, func=residuals):
    """
    Performs multiple fits.
    :param int count: Number of fits to do, each with different
                      noisey data
    :return pd.DataFrame: columns are 'MEAN', 'STD',
                          are species
    """
    def runSim(count, method='leastsq', noise_std=0.5, residuals=residuals):
    # Define the parameters present
    global y_data
    constants = {'k1': [], 'k2': [], 'k3': []}
    for _ in range(count):
        for i in range (nDataPoints):
            y_data[i] = max(data[i, 2] + np.random.normal (0, 5), 0) # standard deviation of noise
        parameters = lmfit.Parameters()
        for constant in constants.keys():
            parameters.add(constant, value=1, min=0, max=10)
        # Create the minimizer
        import pdb; pdb.set_trace()
        fitter = lmfit.Minimizer(residuals, parameters)
        result = fitter.minimize (method=method)
        for constant in constants.keys():
           constants[constant].append(result.params.get(constant).value)
    return {k: [np.mean(v), np.std(v)] for k,v in constants.items()}
        
