'''Cross validation codes.'''

"""
kwargs: keyword arguments to runSimulation
order of positional arguments: obs_data, model, parameters
"""

from collections import namedtuple
import lmfit   # Fitting lib
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import random 
import tellurium as te

TIME = "time"
ME_LEASTSQ = "leastsq"
ME_DIFFERENTIAL_EVOLUTION = "differential_evolution"
ME_BOTH = "both"
PER05 = 0.05
PER95 = 0.95
TIME_INTERVAL = 10

# data - named_array result
# road_runner - road_runner instance created
SimulationResult = namedtuple("SimulationResult",
    "data road_runner")
ResidualCalculation = namedtuple("ResidualCalculation",
    "residuals road_runner")
Statistic = namedtuple("Statistic", "mean std ci_low ci_high")

############## CONSTANTS ######################
# Default simulation model
MODEL = """
     A -> B; k1*A
     B -> C; k2*B
      
     A = 5;
     B = 0;
     C = 0;
     k1 = 0.1
     k2 = 0.2
"""
CONSTANTS = ['k1', 'k2']
NOISE_STD = 0.5
NUM_POINTS = 10
KWARGS_NUM_POINTS = "num_points"
PARAMETERS = lmfit.Parameters()
PARAMETERS.add('k1', value=1, min=0, max=10)
PARAMETERS.add('k2', value=1, min=0, max=10)
ROAD_RUNNER = None
SIM_TIME = 30

# Default parameters values
DF_CONFIDENCE_INTERVAL = (5, 95)
DF_BOOTSTRAP_COUNT = 5
DF_NUM_FOLDS = 3
DF_METHOD = "least_squares"


############## FUNCTIONS ######################
def cleanColumns(df_data, is_force_time=True):
  """
  Cleans the column names in the dataframe,
  removing "[", "]". Makes time index.
  :param pd.DataFrame df_data: Simulation data output.
  :param bool is_force_time: force time to factors of 10
  """
  df = df_data.copy()
  columns = []
  for col in df_data.columns:
    new_col = str(col)
    new_col = new_col.replace("[", "")
    new_col = new_col.replace("]", "")
    columns.append(new_col)
  df.columns = columns
  if TIME in df.columns:
    df = df.set_index(TIME)
  if df.index.name == TIME:
    if is_force_time:
      df.index = [float(v*TIME_INTERVAL) for v in range(len(df))]
      df.index.name = TIME
  return df

def matrixToDF(matrix, columns=None, index=None):
  """
  Converts an array to a dataframe. If the index is
  time, then rounds to a tenth.
  :param np.arrayy matrix:
         pd.DataFrame    : already converted
  :param list index: index for dataframe
  :return pd.DataFrame: Columns are variables w/o [, ].
                        index is time.
  """
  if isinstance(matrix, pd.DataFrame):
    df = cleanColumns(matrix)
  else:
    df = pd.DataFrame(matrix)
    if columns is None:
      try:
        columns = matrix.colnames
      except:
        pass
  if columns is not None:
    df.columns = columns
  return cleanColumns(df)

def matrixToDFWithoutTime(matrix, columns=None):
  """
  Converts an array to a dataframe, deleting time as a column.
  :param np.arrayy matrix:
         pd.DataFrame    : already converted
  :return pd.DataFrame:
  """
  if isinstance(matrix, pd.DataFrame):
    df = matrix.copy()
  else:
    df = matrixToDF(matrix, columns=columns)
  if TIME != df.index.name:
    df = df[df.columns.tolist()[1:]]
  return df

def reshapeData(matrix, indices=None):
  """
  Re-structures matrix as an array for just the rows
  in indices.
  :param array matrix: matrix of data
  :param list-int indices: indexes to include in reshape
  """
  if indices is None:
    nrows = np.shape(matrix)[0]
    indices = range(nrows)
  num_columns = np.shape(matrix)[1]
  trimmed_matrix = matrix[indices, :]
  return np.reshape(trimmed_matrix, num_columns*len(indices))

def arrayDifference(matrix1, matrix2, indices=None):
  """
  Calculates matrix1 - matrix2 as a nX1 array for the rows
  specified in indices.
  """
  array1 = reshapeData(matrix1, indices=indices)
  array2 = reshapeData(matrix2, indices=indices)
  return (array1 - array2)

def makeArrayFromMatrix(df_mat, indices):
  """
  Returns an constructured from specified rows
  organized in sequence by columns.
  :param pd.DataFrame df_mat:
  :param list-int indices:
  :return np.array:
  """
  df = df_mat.loc[df_mat.index[indices], :]
  array = df.T.values
  return np.reshape(array, len(indices)*len(df.columns))

def makeDFWithCommonColumns(df1, df2):
  """
  Returns dataframes that have columns in common.
  """
  columns = set(df1.columns).intersection(df2.columns)
  df1_sub = df1.copy()
  df2_sub = df2.copy()
  df1_sub = df1_sub[columns]
  df2_sub = df2_sub[columns]
  return df1_sub, df2_sub

def calcRsq(observations, estimates, indices=None):
  """
  Computes RSQ for simulation results.
  :param matrix observations: non-time values
  :      pd.DataFrame       : no time column
  :param matrix estimates: non-time values
  :      pd.DataFrame       : no time column
  :param list-int indices:
  :return float:
  """
  df_obs = matrixToDF(observations)
  df_est = matrixToDF(estimates)
  if indices is None:
    indices = range(len(df_obs))
  #
  df_obs_sub, df_est_sub = makeDFWithCommonColumns(df_obs, df_est)
  arr_obs = makeArrayFromMatrix(df_obs_sub, indices)
  arr_est = makeArrayFromMatrix(df_est_sub, indices)
  try:
    arr_rsq = arr_obs - arr_est
  except:
    import pdb; pdb.set_trace()
  arr_rsq = arr_rsq*arr_rsq
  rsq = 1 - sum(arr_rsq) / np.var(arr_obs)
  return rsq

def makeParameters(constants=CONSTANTS, values=1, mins=0, maxs=10):
  """
  Constructs parameters for the constants provided.
  :param list-str constants: names of parameters
  :param list-float values: initial value of parameter
                            if not list, the value for list
  :param list-float mins: minimum values
                            if not list, the value for list
  :param list-float maxs: maximum values
                            if not list, the value for list
  """
  def makeList(val, list_length):
    if isinstance(val, list):
      return val
    else:
      return list(np.repeat(val, list_length))
  # 
  values = makeList(values, len(constants))
  mins = makeList(mins, len(constants))
  maxs = makeList(maxs, len(constants))
  parameters = lmfit.Parameters()
  for idx, constant in enumerate(constants):
    parameters.add(constant,
        value=values[idx], min=mins[idx], max=maxs[idx])
  return parameters

def makeAverageParameters(list_parameters):
  """
  Averages the values of parameters in a list.
  :param list-lmfit.Parameters list_parameters:
  :return lmfit.Parameters: sets min, max to extremes of list
  """
  result_parameters = lmfit.Parameters()
  names = list_parameters[0].valuesdict().keys()
  for name in names:
    values = []
    mins = []
    maxs = []
    for parameters in list_parameters:
      values.append(parameters.valuesdict()[name])
      mins.append(parameters.get(name).min)
      maxs.append(parameters.get(name).max)
    value = np.mean(values)
    min_val = min(mins)
    max_val = min(maxs)
    result_parameters.add(name, value=value, 
        min=min_val, max=max_val)
  return result_parameters

def runSimulation(sim_time=SIM_TIME, 
    num_points=NUM_POINTS, road_runner=ROAD_RUNNER, parameters=None,
    **kwargs):
  """
  Runs the simulation model rr for the parameters.
  :param int sim_time: time to run the simulation
  :param int num_points: number of timepoints simulated
  :param ExtendedRoadRunner road_runner:
  :param lmfit.Parameters parameters:
  :param dict kwargs: parameters used in makeSimulation
  :return SimulationResult:
  """
  if road_runner is None:
    road_runner = makeSimulation(parameters=parameters, **kwargs)
  else:
    road_runner.reset()
    setSimulationParameters(road_runner, parameters)
  data = road_runner.simulate (0, sim_time, num_points)
  simulation_result = SimulationResult(
      data=data,
      road_runner=road_runner,
      )
  return simulation_result

def setSimulationParameters(road_runner, parameters=None):
  """
  Sets parameter values in a road runner instance.
  :param ExtendedRoadRunner road_runner:
  :param lmfit.Parameters parameters:
  """
  if parameters is not None:
    parameter_dict = parameters.valuesdict()
    # Set the simulation constants for all parameters
    for constant in parameter_dict.keys():
      stmt = "road_runner.%s = float(parameter_dict['%s'])" % (
          constant, constant)
      try:
        exec(stmt)
      except:
        import pdb; pdb.set_trace()

def makeSimulation(parameters=None, model=MODEL):
  """
  Creates an road runner instance for the simulation.
  :param lmfit.Parameters parameters:
  :param str model:
  :return ExtendedRoadRunner:
  """
  road_runner = te.loada(model)
  setSimulationParameters(road_runner, parameters)
  return road_runner

def plotTimeSeries(data, is_scatter=False, title="", 
    columns=None, is_plot=True):
  """
  Constructs a time series plot of simulation data.
  :param array data: first column is time
  :param bool is_scatter: do a scatter plot
  :param str title: plot title
  """
  if is_scatter:
    plt.plot (data[:, 0], data[:, 1:], marker='*', linestyle='None')
  else:
    plt.plot (data[:, 0], data[:, 1:])
  plt.title(title)
  plt.xlabel("Time")
  plt.ylabel("Concentration")
  if columns is None:
    columns = np.repeat("", np.shape(data)[1])
  plt.legend(columns)
  if is_plot:
    plt.show() 

def foldGenerator(num_points, num_folds):
  """
  Creates generator for test and training data indices.
  :param int num_points: number of data points
  :param int num_folds: number of folds
  :return iterable: Each iteration produces a tuple
                    First element: training indices
                    Second element: test indices
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

def makeObservations(sim_time=SIM_TIME, num_points=NUM_POINTS,
    noise_std=NOISE_STD, **kwargs):
  """
  Creates synthetic observations.
  :param int sim_time: time to run the simulation
  :param int num_points: number of timepoints simulated
  :param float noise_std: Standard deviation for random noise
  :param dict kwargs: keyword parameters used by runSimulation
  :return pd.DataFrame: simulation results with randomness
  """
  # Create true values
  simulation_result = runSimulation(sim_time=sim_time,
      num_points=num_points,
      **kwargs)
  data = simulation_result.data
  num_cols = len(data.colnames)
  # Add randomness
  for i in range (num_points):
    for j in range(1, num_cols):
      data[i, j] = max(data[i, j]  \
          + np.random.normal(0, noise_std, 1), 0)
  return matrixToDF(data)

def calcSimulationResiduals(obs_data, parameters,
    indices=None, **kwargs):
  """
  Runs a simulation with the specified parameters and calculates residuals
  for the train_indices.
  :param array obs_data: matrix of data, first col is time.
         pd.DataFrame  : index is time
  :param lmfit.Parameters parameters:
  :param list-int indices: indices for which calculation is done
                           if None, then all.
  :param dict kwargs: optional parameters passed to simulation
  :return ResidualCalculation:
  """
  df_obs = matrixToDFWithoutTime(obs_data)
  if indices is None:
    indices = range(len(df_obs))
  if not KWARGS_NUM_POINTS in kwargs.keys():
    kwargs[KWARGS_NUM_POINTS] = len(df_obs)
  simulation_result = runSimulation(parameters=parameters,
      **kwargs)
  df_sim = matrixToDFWithoutTime(simulation_result.data)
  # Compute differences
  df_obs, df_sim = makeDFWithCommonColumns(df_obs, df_sim)
  arr_obs = makeArrayFromMatrix(df_obs, indices)
  arr_sim = makeArrayFromMatrix(df_sim, indices)
  residuals = arr_obs - arr_sim
  residual_calculation = ResidualCalculation(residuals=residuals,
      road_runner=simulation_result.road_runner)
  return residual_calculation

# TODO: Fix handling of obs_data columns. May not be
#       a named array, as in bootstrap.
def fit(obs_data, indices=None, parameters=PARAMETERS, 
    method=ME_LEASTSQ, **kwargs):
  """
  Does a fit of the model to the observations.
  :param ndarray obs_data: matrix of observed values with time
                           as the first column
         pd.DataFrame    : time is index
  :param list-int indices: indices on which fit is performed
  :param lmfit.Parameters parameters: parameters fit
  :param str method: optimization method; both means
                     differential_evolution followed by leastsq
  :param dict kwargs: optional parameters passed to runSimulation
  :return lmfit.Parameters:
  """
  global road_runner
  road_runner = ROAD_RUNNER
  def calcLmfitResiduals(parameters):
    global road_runner
    residual_calculation = calcSimulationResiduals(obs_data,
        parameters, indices, road_runner=road_runner, **kwargs)
    road_runner = residual_calculation.road_runner
    return residual_calculation.residuals
  #
  def estimateParameters(method, parameters):
    # Estimate the parameters for this fold
    fitter = lmfit.Minimizer(calcLmfitResiduals, parameters)
    fitter_result = fitter.minimize(method=method)
    return fitter_result.params
  #
  if method == ME_BOTH:
    parameters = estimateParameters(ME_DIFFERENTIAL_EVOLUTION,
        parameters)
    return estimateParameters(ME_LEASTSQ, parameters)
  else:
    return estimateParameters(method, parameters)

  

def crossValidate(obs_data, method=DF_METHOD,
    sim_time=SIM_TIME,
    num_points=None, parameters=PARAMETERS,
    num_folds=3, **kwargs):
  """
  Performs cross validation on an antimony model.
  :param ndarray obs_data: data to fit; 
                           columns are species; 
                           rows are time instances
                           first column is time
        pd.DataFrame    : time is index
  :param int sim_time: length of simulation run
  :param int num_points: number of time points produced.
  :param lmfit.Parameters: parameters to be estimated
  :param dict kwargs: optional arguments used in simulation
  :return list-lmfit.Parameters, list-float: parameters and RSQ from folds
  """
  df_obs = matrixToDF(obs_data)
  if num_points is None:
    num_points = len(df_obs)
  # Iterate for for folds
  fold_generator = foldGenerator(num_points, num_folds)
  result_parameters = []
  result_rsqs = []
  road_runner = ROAD_RUNNER
  for train_indices, test_indices in fold_generator:
    # This function is defined inside the loop because it references a loop variable
    new_parameters = parameters.copy()
    fitted_parameters = fit(df_obs, method=method,
      indices=train_indices, parameters=new_parameters,
      sim_time=sim_time, num_points=num_points,
      **kwargs)
    result_parameters.append(fitted_parameters)
    # Run the simulation using
    # the parameters estimated using the training data.
    simulation_result = runSimulation(road_runner=road_runner,
        sim_time=sim_time, num_points=num_points,
        parameters=fitted_parameters, **kwargs)
    road_runner = simulation_result.road_runner
    df_est = matrixToDF(simulation_result.data)
    # Calculate RSQ
    rsq = calcRsq(df_obs, df_est, test_indices)
    result_rsqs.append(rsq)
  return result_parameters, result_rsqs

def makeResidualDF(obs_data, model, parameters,
    road_runner=ROAD_RUNNER,
    **kwargs):
  """
  Calculate residuals for each chemical species.
  :param named_array/pd.DataFrame obs_data: matrix of observations; first column is time; number of rows is num_points
  :param lmfit.Parameters parameters:
  :param dict kwargs: optional arguments to runSimulation
  :return pd.DataFrame: matrix of residuals; columns are chemical species
  """
  df_obs = matrixToDF(obs_data)
  if df_obs.index.name != TIME:
    import pdb; pdb.set_trace()
    raise ValueError("Invalid observational data")
  residual_calculation = calcSimulationResiduals(df_obs, parameters,
      model=model, road_runner=road_runner, **kwargs)
  road_runnder = residual_calculation.road_runner
  residuals = residual_calculation.residuals
  # Reshape the residuals by species
  num_species = len(df_obs.columns)
  nrows = int(len(residuals) / num_species)
  result = np.reshape(residuals, (nrows, num_species))
  df_res = pd.DataFrame(result)
  df_res.columns = df_obs.columns
  return df_res

def makeSyntheticObservations(df_res, **kwargs):
  """
  Constructs synthetic observations for the model.
  :param pd.DataFrame df_res: matrix of residuals; columns are species; number of rows is num_points
  :param dict kwargs: optional arguments to runSimulation
  :return pd.DataFrame: index is time
  """
  simulation_result = runSimulation(**kwargs)
  df_data = matrixToDF(simulation_result.data)
  df_data, df_res = makeDFWithCommonColumns(df_data, df_res)
  df_res.columns = df_data.columns
  nrows = len(df_data)
  ncols = len(df_data.columns)
  for col in df_data.columns:
    for idx in df_data.index:
      random_index = df_res.index[np.random.randint(0, nrows)]
      df_data.loc[idx, col] = df_data.loc[idx, col]  \
          + df_res.loc[random_index, col]
  df_data = df_data.applymap(lambda v: max(v, 0))
  return df_data

def doBootstrapWithResiduals(df_res,
    method=DF_METHOD, count=DF_BOOTSTRAP_COUNT, **kwargs):
  """
  Performs bootstrapping by repeatedly generating synthetic observations.
  :param pd.DataFrame residuals_matrix:
  :param int count: number of iterations in bootstrap
  :param dict kwargs: optional arguments to runSimulation
  """
  list_parameters = []
  for _ in range(count):
      df_data = makeSyntheticObservations(df_res, **kwargs)
      parameters = fit(df_data, method=method, **kwargs)
      list_parameters.append(parameters)
  return list_parameters

def doBootstrap(df_data, model, parameters, 
    method=DF_METHOD, count=DF_BOOTSTRAP_COUNT,
    confidence_limits=DF_CONFIDENCE_INTERVAL, **kwargs):
  """
  Performs bootstrapping by repeatedly generating synthetic observations,
  calculating the residuals as well.
  :param pd.DataFrame df_data: index is time
  :param str model:
  :param lmfit.Parameters parameters:
  :param int count: number of iterations in bootstrap
  :param dict kwargs: optional arguments to runSimulation
  :return dict: confidence limits
  """
  df_res = makeResidualDF(df_data, model,
      parameters, **kwargs)
  list_parameters = doBootstrapWithResiduals(df_res,
      method=method,
      count=count, model=model, parameters=parameters, **kwargs)
  return makeParameterStatistics(list_parameters,
      confidence_limits=confidence_limits)

def calcStatistic(values, confidence_limits=[PER05, PER95]):
  """
  Calculates Statistic
  :param list-float values:
  :return Statistic:
  """
  quantiles = np.quantile(values, confidence_limits)
  return Statistic(
      mean  =np.mean(values),
      std = np.std(values),
      ci_low = quantiles[0],
      ci_high = quantiles[1],
      )

def makeParameterStatistics(list_parameters,
    confidence_limits=DF_CONFIDENCE_INTERVAL):
  """
  Computes the mean and standard deviation of the parameters in a list of parameters.
  :param list-lmfit.Parameters
  :param (float, float) confidence_limits: if none, report mean and variance
  :return dict: key is the parameter name; value is ParameterStatistic
  """
  statistic_dict = {}
  parameter_names = list(list_parameters[0].valuesdict().keys())
  for name in parameter_names:
    statistic_dict[name] = []
    for parameters in list_parameters:
      statistic_dict[name].append(parameters.valuesdict()[name])
  # Calculate the statistics
  for name in statistic_dict.keys():
    statistic_dict[name] = calcStatistic(statistic_dict[name])
  return statistic_dict
