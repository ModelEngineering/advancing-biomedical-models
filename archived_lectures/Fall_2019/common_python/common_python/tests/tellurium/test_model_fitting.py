'''Tests for model fitting.'''

from common_python.tellurium import model_fitting as mf

import lmfit
import numpy as np

############ CONSTANTS #############
IS_PLOT = False
NROWS = 10
NROWS_SUBSET = 5
NCOLS = 3
LENGTH = NROWS*NCOLS
INDICES = range(NROWS)
# Set to values used in mf.MODEL
TEST_PARAMETERS = lmfit.Parameters()
TEST_PARAMETERS.add('k1', value=0.1, min=0, max=10)
TEST_PARAMETERS.add('k2', value=0.2, min=0, max=10)
CSV_FILE = "wild.csv"

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
  array = mf.reshapeData(data, range(NROWS_SUBSET))
  assert(len(array) == NROWS_SUBSET*NCOLS)
  assert(np.shape(array)[0] == len(array))

def testArrayDifference():
  matrix1 =  makeData(NROWS, NCOLS)
  matrix2 =  makeData(NROWS, NCOLS)
  array = mf.arrayDifference(matrix1, matrix2, INDICES)
  assert(sum(np.abs(array)) == 0)

def testCalcRsq():
  std = 0.5
  matrix = np.reshape(np.random.normal(0, std, LENGTH),
      (NROWS, NCOLS))
  rsq = mf.calcRsq(matrix, matrix)
  assert(np.isclose(rsq, 1))
  #
  matrix2 = np.reshape(np.random.normal(0, std, LENGTH),
      (NROWS, NCOLS))
  rsq = mf.calcRsq(matrix, matrix2)
  assert(rsq < 0.01)

def testMakeParameters():
  constants =  ['k1', 'k2', 'k3']
  parameters = mf.makeParameters(constants=constants)
  assert(len(parameters.valuesdict()) == len(constants))

def testMakeAverageParameters():
  """
  Constructs parameter values that are the average of existing parameters.
  """
  list_parameters = [TEST_PARAMETERS, TEST_PARAMETERS]
  average_parameters = mf.makeAverageParameters(
      list_parameters)
  test_dict = TEST_PARAMETERS.valuesdict()
  result_dict = average_parameters.valuesdict()
  for name in test_dict.keys():
    assert(test_dict[name] == result_dict[name])

def testRunSimulation():
  simulation_result1 = mf.runSimulation() 
  assert(simulation_result1.data[-1, 0] == mf.SIM_TIME)
  simulation_result2 = mf.runSimulation(parameters=TEST_PARAMETERS) 
  nrows, ncols = np.shape(simulation_result1.data)
  for i in range(nrows):
    for j in range(ncols):
      assert(np.isclose(simulation_result1.data[i,j],
           simulation_result2.data[i,j]))

def testPlotTimeSeries():
  # Smoke test only
  simulation_result = mf.runSimulation() 
  mf.plotTimeSeries(simulation_result.data, is_plot=IS_PLOT)
  mf.plotTimeSeries(simulation_result.data, 
      is_scatter=True, is_plot=IS_PLOT)
  

def testMakeObservations():
  def test(num_points):
    df_obs = mf.makeObservations(
        num_points=num_points,
        road_runner=mf.ROAD_RUNNER)
    simulation_result  = mf.runSimulation(
        num_points=num_points,
        road_runner=mf.ROAD_RUNNER)
    df_sim = mf.matrixToDF(simulation_result.data)
    assert(len(df_obs) == len(df_sim))
    df_res = df_obs - df_sim
    ser_std = df_res.std()
    trues = [v < 3*mf.NOISE_STD for v in ser_std.values]
    assert(all(trues))
  #
  test(mf.NUM_POINTS)
  test(2*mf.NUM_POINTS)

def testCalcSimulationResiduals():
  def test(data):
    residual_calculation = mf.calcSimulationResiduals(data,
        TEST_PARAMETERS)
    residuals = residual_calculation.residuals
    assert(sum(residuals*residuals) == 0)
  #
  simulation_result = mf.runSimulation(parameters=TEST_PARAMETERS)
  matrix = simulation_result.data
  test(matrix)
  df = mf.matrixToDF(matrix)
  test(df)

def testFit():
  obs_data = mf.makeObservations()
  def test(method=mf.ME_LEASTSQ):
    parameters = mf.fit(obs_data, method=method)
    param_dict = dict(parameters.valuesdict())
    expected_param_dict = dict(mf.PARAMETERS.valuesdict())
    diff = set(param_dict.keys()).symmetric_difference(
        expected_param_dict.keys())
    assert(len(diff) == 0)
  #
  test()
  test(mf.ME_BOTH)

def testCrossValidate():
  def test(data, min_rsq):
    results_parameters, results_rsqs = mf.crossValidate(data)
    parameters_avg = mf.makeAverageParameters(
        results_parameters)
    params_dict = parameters_avg.valuesdict()
    for name in params_dict.keys():
      assert(np.abs(params_dict[name]  \
      - TEST_PARAMETERS.valuesdict()[name]) < 2*params_dict[name])
    for rsq in results_rsqs:
      assert(rsq >= min_rsq)
  #
  obs_data = mf.makeObservations(model=mf.MODEL, noise_std=0)
  test(obs_data, 1)
  test(mf.matrixToDF(obs_data), 1)
  obs_data = mf.makeObservations(model=mf.MODEL, noise_std=0.1)
  test(obs_data, 0.7)

def testCrossValidate2():
  num_points = 20
  obs_data = mf.makeObservations(
      parameters=TEST_PARAMETERS, num_points=num_points)
  results_parameters, results_rsq = mf.crossValidate(
      obs_data, num_points=num_points, num_folds=10)
  parameters_avg = mf.makeAverageParameters(
      results_parameters)
  params_dict = parameters_avg.valuesdict()
  for name in params_dict.keys():
    assert(np.abs(params_dict[name]  \
    - TEST_PARAMETERS.valuesdict()[name]) < 2*params_dict[name])

def testMakeResidualDF():
  num_points = 20
  max_val = 10 
  df_res = _getResiduals(num_points)
  assert(len(df_res) == num_points)
  assert(df_res.sum().sum() < max_val)

def testMakeSyntheticObservations():
  num_points = 20
  kwargs = {'model': mf.MODEL,
            'num_points': num_points,
           }
  df_res = _getResiduals(num_points, model=kwargs['model'])
  df_syn = mf.makeSyntheticObservations(df_res, **kwargs)
  assert(len(df_syn) == num_points)
  assert(len(set(df_syn.columns).symmetric_difference(
       df_res.columns)) == 0)

def _makeParameterList(count, num_points=mf.NUM_POINTS, **kwargs):
  residual_matrix = _getResiduals(num_points, model=kwargs['model'])
  return mf.doBootstrapWithResiduals(residual_matrix, count=count,
      num_points=num_points,
      **kwargs)

def testDoBootstrapWithResiduals():
  num_points = 20
  count = 3
  list_parameters = _makeParameterList(count, num_points,
      model=mf.MODEL)
  assert(len(list_parameters) == count)
  for parameters in list_parameters:
    assert(isinstance(parameters, lmfit.Parameters))

def _getResiduals(num_points, model=mf.MODEL):
  df_obs = mf.makeObservations(model=model,
      parameters=TEST_PARAMETERS, num_points=num_points)
  return mf.makeResidualDF(df_obs, model,
      TEST_PARAMETERS, num_points=num_points)

def testDoBootstrap():
  num_points = 20
  count = 3
  model = mf.MODEL
  df_obs = mf.makeObservations(model=model,
      parameters=TEST_PARAMETERS, num_points=num_points)
  statistic_dict = mf.doBootstrap(df_obs, model,
      TEST_PARAMETERS, count=count,
      num_points=num_points)
  params_dict = TEST_PARAMETERS.valuesdict()
  diff = set(params_dict.keys()).symmetric_difference(
      statistic_dict.keys())
  assert(len(diff) == 0)

def testDoBootstrap2():
  model0 = """
       # True model
       A  -> B + D; k1*A
       B -> D; k2*B
       D -> C; k3*A*B
        
       A = 5;
       B = 0;
       C = 0;
       D = 0;
       k1 = 0.08
       k2 = 0.1
       k3 = 0.1
  """
  num_points = 20
  sim_time = 20
  unfitted_parameters = mf.makeParameters(
      constants=['k1', 'k2', 'k3'])
  df_full_obs = mf.makeObservations(model=model0, 
      noise_std=0.3, num_points=num_points, sim_time=sim_time)
  result = mf.doBootstrap(df_full_obs, 
      model=model0, parameters=unfitted_parameters, 
      num_points=num_points, sim_time=sim_time, count=5)
 
def testMakeParameterStatistics():
  num_points = 20
  count = 3
  list_parameters = _makeParameterList(count, num_points,
      model=mf.MODEL)
  def test(confidence_limits):
    statistic_dict = mf.makeParameterStatistics(list_parameters,
        confidence_limits)
    for key in statistic_dict.keys():
      assert(isinstance(statistic_dict[key], mf.Statistic))
  #
  test((5, 95))
  test(None)

def testMatrixToDF():
  size = 10
  ncol = 2
  colnames = [mf.TIME, 'B']
  def test(columns=None):
    matrix = np.reshape(list(range(size)), (int(size/ncol), ncol))
    df = mf.matrixToDF(matrix, columns=colnames)
    assert(len(df) == int(size/ncol))
    if columns is not None:
      columns = list(set(columns).difference([mf.TIME]))
      try:
        assert(len(
          set(columns).symmetric_difference(df.columns)) == 0)
      except:
        import pdb; pdb.set_trace()
  #
  test()
  test(columns=colnames)

def testMatrixToDFWithoutTime():
  size = 10
  ncol = 2
  colnames = [mf.TIME, 'B']
  def test(columns=None):
    matrix = np.reshape(list(range(size)), (int(size/ncol), ncol))
    df = mf.matrixToDFWithoutTime(matrix, columns=columns)
    assert(len(df) == int(size/ncol))
    assert(len(df.columns) == ncol - 1)
    if columns is not None:
      assert(len(
          set(columns).symmetric_difference(df.columns)) == 1)
  #
  test(columns=colnames)
  test()

def testCalcStatistic():
  values = range(100)
  statistic = mf.calcStatistic(values)
  assert((statistic.ci_low < statistic.mean)  
      and (statistic.mean < statistic.ci_high))
   
  
if __name__ == '__main__':
  testMakeResidualDF()
  if True:
    testReshapeData() 
    testArrayDifference() 
    testCalcRsq()
    testMakeParameters()
    testMakeAverageParameters()
    testRunSimulation()
    testPlotTimeSeries()
    testCalcSimulationResiduals()
    testFit()
    testCrossValidate()
    testMakeObservations()
    testCrossValidate2()
    testMakeResidualDF()
    testMakeSyntheticObservations()
    testDoBootstrapWithResiduals()
    testDoBootstrap()
    testDoBootstrap2()
    testMakeParameterStatistics()
    testMatrixToDF()
    testMatrixToDFWithoutTime()
    testCalcStatistic()
  print("OK")
