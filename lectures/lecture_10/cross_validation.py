'''Cross validation codes.'''

import tellurium as te
import numpy as np
import lmfit   # Fitting lib
import math
import random 



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
SIM_TIME = 30
NUM_POINTS = 10
CONSTANTS = ['k1', 'k2']
ROAD_RUNNER = te.loada(MODEL)


############## FUNCTIONS ######################
def reshapeData(matrix, indices, first_col=0):
    """
    Re-structures matrix as an array for just the rows
    in indices.
    """
    num_columns = np.shape(matrix)[1] - first_col
    trimmed_matrix = matrix[indices, first_col:]
    return np.reshape(trimmed_matrix, num_columns*len(indices))

def arrayDifference(matrix1, matrix2, indices):
    """
    Calculates matrix1 - matrix2 as a nX1 array for the rows
    specified in indices.
    """
    array1 = reshapeData(matrix1, indices)
    array2 = reshapeData(matrix2, indices)
    return (array1 - array2)

def calcRsq(observations, estimates, indices):
    """
    Computes RSQ for simulation results.
    :param 2d-np.array observations: non-time values
    :param 2d-np.array estimates: non-time values
    :param list-int indices:
    :return float:
    """
    array_residuals = arrayDifference(observations, estimates, indices)
    array_observations = reshapeData(observations, indices)
    return 1 - np.var(array_residuals)/np.var(array_observations)

def runSimulation(sim_time, num_points, parameters=None,
    road_runner=None, model=None):
    """
    Runs the simulation model rr for the parameters.
    :param int sim_time: time to run the simulation
    :param int num_points: number of timepoints simulated
    :param lmfit.Parameters parameters:
    :param ExtendedRoadRunner road_runner:
    :param str model:
    :return named_array:
    """
    if road_runner is None:
        road_runner = te.loada(model)
    else:
        road_runner.reset()
    if parameters is not None:
        parameter_dict = parameters.valuesdict()
        # Set the simulation constants for all parameters
        for constant in parameter_dict.keys():
            stmt = "road_runner.%s = parameter_dict['%s']" % (constant, constant)
            exec(stmt)
    return road_runner.simulate (0, sim_time, num_points)

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

def makeObservations(sim_time, num_points, noise_std=0.5, **kwargs):
    """
    Creates synthetic observations.
    :param int sim_time: time to run the simulation
    :param int num_points: number of timepoints simulated
    :param float noise_std: Standard deviation for random noise
    :param dict kwargs: keyword parameters used by runSimulation
    :return namedarray: simulation results with randomness
    """
    # Create true values
    data = runSimulation(sim_time, num_points, **kwargs)
    num_cols = len(data.colnames) - 1
    # Add randomness
    for i in range (num_points):
        for j in range(1, num_cols):
            data[i, j] = max(obs_data[i, j] + np.random.normal(0, noise_std, 1), 0)
    return data

def fit(obs_data, model=MODEL, indices=None, parameters=PARAMETERS,
      sim_time=SIM_TIME, num_points=NUM_POINTS, method='leastsq',
      road_runner=ROAD_RUNNER):
    """
    Does a fit of the model to the observations.
    :return lmfit.Parameters:
    """
    def calcResiduals(parameters):
        """
        Runs a simulation with the specified parameters and calculates residuals
        for the train_indices.
        """
        sim_data = runSimulation(sim_time, num_points, road_runner=rr, parameters=parameters)
        sim_data = sim_data[:, 1:]  # Skip time
        return arrayDifference(obs_data, sim_data, indices)
    # Estimate the parameters for this fold
    fitter = lmfit.Minimizer(calcResiduals, parameters)
    fitter_result = fitter.minimize(method=method)
    return fitter_result.params

def makeParameters(constants=CONSTANTS):
    parameters = lmfit.Parameters()
    for constant in constants:
      parameters.add(constant, value=1, min=0, max=10)
    return parameters

def cross_validate(obs_data, model=MODEL, sim_time=SIM_TIME, num_points=NUM_POINTS,
        constants=CONSTANTS;
        noise_std=0.5, num_folds=3):
    """
    Performs cross validation on an antimony model.
    :param ndarray obs_data: data to fit; columns are species; rows are time instances
    :param str model: antimony model
    :param int sim_time: length of simulation run
    :param int num_points: number of time points produced.
    :param list-str constants: constants to be estimated
    :return list-lmfit.Parameters, list-float: parameters and RSQ from folds
    """
    # Iterate for for folds
    fold_generator = foldGenerator(num_points, num_folds)  # Create the iterator object
    result_parameters = []
    result_rsqs = []
    parameters = makeParameters(constants=constants)
    for train_indices, test_indices in fold_generator:
        # This function is defined inside the loop because it references a loop variable
        parameters = makeParameters(constants=constants)
        fitter_result = fit(obs_data, model=MODEL, indices=None, parameters=parameters,
            sim_time=SIM_TIME, num_points=NUM_POINTS, road_runner=ROAD_RUNNER)
        result_parameters.append(fitter_result.params)
        # Run the simulation using
        # the parameters estimated using the training data.
        test_estimates = runSimulation(sim_time, num_points, road_runner=rr,
              model-model, parameters=fitter_result.params)
        test_estimates = test_estimates[:, 1:]
        # Calculate RSQ
        rsq = calcRsq(obs_data, test_estimates, test_indices)
        result_rsqs.append(rsq)
    return result_parameters, result_rsqs
