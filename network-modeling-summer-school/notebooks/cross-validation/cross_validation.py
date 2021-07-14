#!/usr/bin/env python
# coding: utf-8

# # CROSS VALIDATION: IMPLEMENTATION
# 
# This notebook describes how to do cross validation for simulation models.
# 
# Cross validation partitions the observed data into *folds*.
# A fold consists of a pair of training data and test data.
# For simulation models,
# the training data are used to estimate parameters, and the quality of these estimates is evaluated based on predictions of the test data.
# 
# The main challenges with cross validation for simulation data are: creating folds, estimating parameters on data subsets, and evaluating parameter estimates on a data subset.

# # Programming Preliminaries

print("# In[41]:")


IS_COLAB = False


print("# In[42]:")


if IS_COLAB:
    get_ipython().system('pip install -q tellurium')
    get_ipython().system('pip install -q SBstoat')


print("# In[43]:")


#get_ipython().run_line_magic('matplotlib', 'inline')
import tellurium as te
import numpy as np
import math
import random 
import matplotlib.pyplot as plt
import urllib
from SBstoat import ModelFitter, NamedTimeseries, Parameter


print("# In[44]:")


def getSharedCodes(moduleName):
  """
  Obtains common codes from the github repository.

  Parameters
  ----------
  moduleName: str
      name of the python module in the src directory
  """
  if IS_COLAB:
      url = "https://github.com/sys-bio/network-modeling-summer-school-2021/raw/main/src/%s.py" % moduleName
      local_python = "python.py"
      _, _ = urllib.request.urlretrieve(url=url, filename=local_python)
  else:
      local_python = "../src/%s.py" % moduleName
  with open(local_python, "r") as fd:
    codeStr = "".join(fd.readlines())
  print(codeStr)
  exec(codeStr, globals())

# Acquire codes
getSharedCodes("util")

# TESTS
assert(isinstance(LINEAR_PATHWAY_DF, pd.DataFrame))


print("# In[45]:")


def findCloseMatchingValues(longArr, shortArr):
    """
    Finds the indices in longArr that are closest to the values in shortArr.

    Parameters
    ----------
    longArr: np.array
    shortArr: np.arry

    Returns
    -------
    array-int
    """
    indices = []
    for val in shortArr:
        distances = (longArr - val)**2
        minDistance = np.min(distances)
        distancesLst = list(distances)
        idx = distancesLst.index(minDistance)
        indices.append(idx)
    return np.array(indices)

# TESTS
longArr = np.array(range(10))
shortArr = np.array([2.1, 2.9, 4.3])
indexArr = findCloseMatchingValues(longArr, shortArr)
expectedArr = [2, 3, 4]
assert(all([v1 == v2 for v1, v2 in zip(indexArr, expectedArr)]))


print("# In[46]:")


# Experimental conditions
END_TIME = max(LINEAR_PATHWAY_ARR[:, 0])
NUM_POINT = 15
NOISE_STD = 1.0
NUM_FOLD = 3


# # Constants

print("# In[47]:")


# Constants
FOLD_TRAINING = 0 # Training data in a fold
FOLD_TEST = 1 # Test data in a fold
RSQ = "rsq" # R-squared value in a dataframe


# # Model

print("# In[48]:")


print(LINEAR_PATHWAY_MODEL)


# # Helper Functions

# The following functions provide capabilities used in this notebook.

print("# In[49]:")


def runSimulation(simTime=END_TIME, numPoint=NUM_POINT, roadRunner=None,
                  parameterDct=None, model=LINEAR_PATHWAY_MODEL):
    """
    Runs the simulation model for the parameters.
   
    Parameters
    ----------
    endTime: float
        End time for the simulation
    numPoints: int
        Number of points in the simulation
    roadRunner: ExtendedRoadRunner
    parameters: list-str
        
    Returns
    -------
    NamedArray
        results of simulation
    """
    if roadRunner is None:
        roadRunner = te.loada(model)
    else:
        roadRunner.reset()
    if parameterDct is not None:
        # Set the simulation constants for all parameters
        for name in parameterDct.keys():
            roadRunner[name] = parameterDct[name]
    return roadRunner.simulate (0, simTime, numPoint)

# TESTS
numPoint = int(10*END_TIME)
fittedData = runSimulation(parameterDct={"k1": 0.1}, numPoint=numPoint)
numCol = np.shape(fittedData)[1]
assert(np.size(fittedData) == numPoint*numCol)
assert(fittedData[0, 1] == 10)


print("# In[50]:")


FITTED_DATA = runSimulation(parameterDct={"k1": 0.1}, numPoint=numPoint)


# # Cross Validation Algorithm

# Cross validation divides the data into training and test sets.
# The training sets are used to estimate parameters.
# The test sets are used to evaluate the quality of the
# parameter fits.
# That is, for each set of test indices, we have a companion set of training indices. The training indices select a subset of the observational data that are used to estimate parameter values.
# 
# Below is pseudo code that calculates the $R^2$ value for each fold.
# 
#     def crossValidate(model, observedData, numFold):
#         folds = generateFolds(observedData, numFold)
#         foldQualities = []
#         for fold folds:
#             foldQuality = evaluateFold(model, fold)
#             foldQualities.append(foldQuality)
#         return foldQualities

# # Generating Folds

# The first step is to generate the folds.
# A fold is represented by a tuple.
# The first element is the training data;
# the second element is the test data.
# The collection of these tuples is itsself an array.
# ``generateFolds`` such an array.
# 
# 

print("# In[51]:")


def generateFolds(observedData, numFold):
    """
    Generates indices of training and test data
    by alternating between folds
    
    Parameters:
    ----------
    observedData: np.array(N, M) or NamedTimeseries
    numFold: int
        number of pairs of testIndices and trainIndices
    
    Returns:
    --------
    array-tuple(array, array)
    """
    if isinstance(observedData, NamedTimeseries):
        df = observedData.to_dataframe()
        df = df.reset_index()
        observedData = df.to_numpy()
    result = []
    numPoint, numCol = np.shape(observedData)
    indices = range(numPoint)
    for remainder in range(numFold):
        testIndices = np.array([n for n in indices if n % numFold == remainder])
        testData = observedData[testIndices, :]
        trainIndices = np.array(list(set(indices).difference(testIndices)))
        trainData = observedData[trainIndices, :]
        entry = (trainData, testData)
        result.append(entry)
    return result

# TESTS
numFold = 3
observedData = NamedTimeseries(dataframe=LINEAR_PATHWAY_DF)
folds = generateFolds(observedData.to_dataframe().to_numpy(), numFold)
assert(len(folds) == numFold)
fold = folds[0]
assert(len(fold) == 2)
trainData = fold[0]
testData = fold[1]
assert(len(observedData) == (len(trainData) + len(testData)))
#
fold2s = generateFolds(observedData, numFold)
assert(len(folds) == len(fold2s))


print("# In[52]:")


folds


print("# In[53]:")


trainData


print("# In[54]:")


testData


# # Evaluating Folds

print("# In[55]:")


def evaluateFold(model, colnames, parametersToFit, fold,
                 **fittingArgs):
    """
    Calculates the R-squared value for the fit of the predicted test data,
    whose parameters are estimated from the training data, with the observed
    tests data.

    Parameters
    ----------
    model: antimony/ExtendedRoadRunner
    colnames: list-str
        names of data columns
    fold: tuple(np.array, np.array)
        train data, test data
    fittingArgs: dict
        optional arguments for ModelFitter

    Returns
    -------
    float: R squared
    ModelFitter
    """
    # Estimate the parameters
    observedTS = NamedTimeseries(colnames=colnames, array=fold[FOLD_TRAINING])
    fitter = ModelFitter(modelSpecification=model,
                                    parametersToFit=parametersToFit,
                                    observedData=observedTS,
                                    **fittingArgs,
                                    )
    fitter.fitModel()
    parameterDct = dict(fitter.params.valuesdict())
    # Obtain the fitted values for the estimated parameters
    if "endTime" in fittingArgs:
        endTime = fittingArgs["endTime"]
    else:
        endTime = END_TIME
    numPoint = int(10*endTime)
    fittedData = runSimulation(simTime=endTime, numPoint=numPoint,
                                parameterDct=parameterDct,
                                roadRunner=model)
    # Find the time indices that correspond to the test data
    testData = fold[FOLD_TEST]
    testTimes = testData[:, 0]
    fittedTimes = fittedData[:, 0]
    indices = findCloseMatchingValues(fittedTimes, testTimes)
    # Calculate residuals for the corresponding times
    indexArr = np.array(indices)
    fittedTestData = fittedData[indexArr, 1:]
    flatFittedTestData = fittedTestData.flatten()
    flatTestData = (testData[:, 1:]).flatten()
    residualsArr = flatTestData - flatFittedTestData
    rsq = 1 - np.var(residualsArr)/np.var(flatTestData)
    #
    return rsq, fitter

# TESTS
model = te.loada(LINEAR_PATHWAY_MODEL)
colnames = list(LINEAR_PATHWAY_DF.columns)
parametersToFit = [
                Parameter("k1", lower=0, value=0, upper=10),
                Parameter("k2", lower=0, value=0, upper=10),
                Parameter("k3", lower=0, value=0, upper=10),
                Parameter("k4", lower=0, value=0, upper=10),
    ]
folds = generateFolds(LINEAR_PATHWAY_ARR, NUM_FOLD)
rsq, fitter = evaluateFold(model, colnames, parametersToFit, folds[0],
                   fitterMethods=["differential_evolution"])
assert(rsq > 0.85)
assert(isinstance(fitter, ModelFitter))


# # Complete Workflow

print("# In[56]:")


def crossValidate(model, observedData, parametersToFit, colnames, numFold,
                  **fitterArgs):
    """
    Performs cross validation on the model.

    Parameters
    ----------
    model: ExtendedRoadrunner
    observedData: NamedTimeseries
    parametersToFit: list-SBstoat.Parameter
    colnames: list-str
    numFold: int

    Results
    -------
    pd.DataFrame
        Index: fold
        Columns
            R2: R squared value
            values of parameters
    """
    folds = generateFolds(observedData, numFold)
    result = {p.name: [] for p in parametersToFit}
    result[RSQ] = []
    for fold in folds:
        foldQuality, fitter = evaluateFold(model, colnames, parametersToFit,
                                           fold, **fitterArgs)
        result[RSQ].append(foldQuality)
        valueDct = fitter.params.valuesdict()
        for parameter in parametersToFit:
            result[parameter.name].append(valueDct[parameter.name])
    df = pd.DataFrame(result)
    return df

# TESTS
model = te.loada(LINEAR_PATHWAY_MODEL)
colnames = list(LINEAR_PATHWAY_DF.columns)
parametersToFit = [
                Parameter("k1", lower=0, value=0, upper=10),
                Parameter("k2", lower=0, value=0, upper=10),
                Parameter("k3", lower=0, value=0, upper=10),
                Parameter("k4", lower=0, value=0, upper=10),
    ]
observedData = NamedTimeseries(dataframe=LINEAR_PATHWAY_DF)
resultDF = crossValidate(model, observedData, parametersToFit,
                          colnames, NUM_FOLD,
                          fitterMethods=["differential_evolution"])
trues = [q > 0.85 for q in resultDF[RSQ]]
assert(all(trues))


print("# In[57]:")


resultDF


# # Exercises

# This exerise uses the ``WOLF_MODEL`` and the data ``WOLF_DF``.
# Only fit the parameters ``J1_k1``, ``J1_ki``, and ``J1_n``.
# 
# 1. Do 2, 10, 20, and 100 fold cross validations. Save the fold qualities in a python dictionary.
# 
# 1. Some questions about the results in (1).
#    1. How does the mean value of quality change with the number of folds?
# 
#    1. Can you explain the trend in the standard deviations of the quality scores?
# 
# 1. Plot the distribution of quality scores for 20 and 100 folds.
# What can you say about these distributions?
#    1. Hint: If ``resultDF`` is the DataFrame returned by ``crossvalidate``,
#    then ``list(resultDF["R2"])`` is the list of $R^2$ values for the folds.
# 

# ## (1) Calculate Cross Validations

print("# In[58]:")


model = te.loada(WOLF_MODEL)
colnames = list(WOLF_DF.columns)
observedData = NamedTimeseries(dataframe=WOLF_DF)


print("# In[59]:")


upper = 1e5
parametersToFit = [
      Parameter("J1_k1", lower=0, value=1, upper=upper), #550
      Parameter("J1_Ki", lower=0, value=1, upper=upper), #1
      Parameter("J1_n", lower=0, value=1, upper=upper), #4                                                                                                                                                                 
]


print("# In[ ]:")


resultDct = {}
for numFold in [2, 10, 20, 100]:
    qualityDF = crossValidate(model, observedData, parametersToFit,
                          colnames, numFold,
                          fitterMethods=["differential_evolution"],
                          endTime=5)
    resultDct[numFold] = qualityDF


print("# In[ ]:")


result[100]


# ## (2) Questions

# 1. The mean values of $R^2$ change little as the folds increase, although there is a small decrease at 100 folds.
# 1. The standard deviation of $R^2$ increases with the number of folds.
# This is because with fewer data values in the test data,
# there is more variability in how well the model fits.

# ## (3) Plot the distributions

print("# In[ ]:")


def plotHist(numFold):
    """
    Plots a histogram for the number of folds.

    Parameters
    ----------
    numFold: int
    """
    _ = plt.hist(resultDct[numFold]["R2"])
    plt.xlim([0, 1.0])
    plt.title("Folds: %d" % numFold)


print("# In[ ]:")


plotHist(20)


print("# In[ ]:")


plotHist(100)


# The histogram plots are consistent with the observation that variance increases with the number of folds.
