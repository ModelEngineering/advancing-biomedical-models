'''Tests for model fitting.'''

import model_fitting

import lmfit
import numpy as np

############ CONSTANTS #############
NROWS = 10
NROWS_SUBSET = 5
NCOLS = 3
LENGTH = NROWS*NCOLS
INDICES = range(NROWS)
# Set to values used in model_fitting.MODEL
TEST_PARAMETERS = lmfit.Parameters()
TEST_PARAMETERS.add('k1', value=0.1, min=0, max=10)
TEST_PARAMETERS.add('k2', value=0.2, min=0, max=10)

def makeData(nrows, ncols, valFunc=None):
  """
  Creates an array in the desired shape.
  :param int nrows:
  :param int ncols:
  :param Function value: Function
                      argument: number of values
                      return: iterable of values
  """
  length = nrows*ncols
  if valFunc is None:
    valFunc = lambda v: range(v)
  data = valFunc(length)
  matrix = np.reshape(data, (nrows, ncols))
  return matrix
  

def testReshapeData():
  data = makeData(NROWS, NCOLS)
  array = model_fitting.reshapeData(data, range(NROWS_SUBSET))
  assert(len(array) == NROWS_SUBSET*NCOLS)
  assert(np.shape(array)[0] == len(array))

def testArrayDifference():
  matrix1 =  makeData(NROWS, NCOLS)
  matrix2 =  makeData(NROWS, NCOLS)
  array = model_fitting.arrayDifference(matrix1, matrix2, INDICES)
  assert(sum(np.abs(array)) == 0)

def testCalcRsq():
  std = 0.5
  residuals = np.reshape(np.random.normal(0, std, LENGTH),
      (NROWS, NCOLS))
  matrix1 =  makeData(NROWS, NCOLS)
  matrix2 = matrix1 + residuals
  rsq = model_fitting.calcRsq(matrix2, matrix1)
  var_est = (1 - rsq)*np.var(matrix1)
  var_exp = std*std
  assert(np.abs(var_est - var_exp) < 0.5)

def testMakeParameters():
  parameters = model_fitting.makeParameters(model_fitting.CONSTANTS)
  assert(len(parameters.valuesdict()) == len(model_fitting.CONSTANTS))
 
def testRunSimulation():
  data1 = model_fitting.runSimulation() 
  assert(data1[-1, 0] == model_fitting.SIM_TIME)
  data2 = model_fitting.runSimulation(
      parameters=TEST_PARAMETERS) 
  nrows, ncols = np.shape(data1)
  for i in range(nrows):
    for j in range(ncols):
      assert(np.isclose(data1[i,j], data2[i,j]))

def testMakeObservations():
  obs_data = model_fitting.makeObservations(
      num_points=model_fitting.NUM_POINTS,
      road_runner=model_fitting.ROAD_RUNNER)
  obs_data = obs_data[:, 1:]
  data = model_fitting.runSimulation(
      num_points=model_fitting.NUM_POINTS,
      road_runner=model_fitting.ROAD_RUNNER)
  data = data[:, 1:]
  nrows, _ = np.shape(data)
  assert(nrows == model_fitting.NUM_POINTS)
  std = np.sqrt(np.var(model_fitting.arrayDifference(obs_data, data)))
  assert(std < 3*model_fitting.NOISE_STD)
  assert(std > model_fitting.NOISE_STD/3.0)

def testFit():
  obs_data = model_fitting.makeObservations()
  parameters = model_fitting.fit(obs_data[:, 1:])
  param_dict = dict(parameters.valuesdict())
  expected_param_dict = dict(model_fitting.PARAMETERS.valuesdict())
  diff = set(param_dict.keys()).symmetric_difference(
      expected_param_dict.keys())
  assert(len(diff) == 0)
  
  
if __name__ == '__main__':
  testReshapeData() 
  testArrayDifference() 
  testCalcRsq()
  testMakeParameters()
  testRunSimulation()
  testMakeObservations()
  testFit()
  print("OK")
