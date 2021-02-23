#!/usr/bin/env python
# coding: utf-8

"""Functions and Classes used for cross validation."""

import copy
import numpy as np
import pandas as pd
import tellurium as te
import matplotlib.pyplot as plt
from SBstoat.observationSynthesizer import ObservationSynthesizerRandomErrors
from SBstoat.namedTimeseries import NamedTimeseries, TIME
import lmfit


# A simple model
MODEL = """
A -> B; k1*A
B -> C; k2*B
C -> D; k3*C

A = 10
B = 0
C = 0
D = 0
k1 = 0.1
k2 = 0.2
k3 = 0.3
"""


NOISE_STD = 1.0
END_TIME = 20
NUM_POINT = 100
LOWER = 0.01
UPPER = 1
# Model data
PARAMETER_DCT = {
    "k1": 0.1,
    "k2": 0.2,
    "k3": 0.3,
}
PARAMETER_NAMES = list(PARAMETER_DCT.keys())
ROAD_RUNNER = te.loada(MODEL)
dataArr = ROAD_RUNNER.simulate(0, END_TIME, NUM_POINT)
FITTED_TS = NamedTimeseries(namedArray=dataArr)


def foldGenerator(numPoint, numFold):
    """
    Generates pairs of trainining and test indices.
    
    Parameters:
    ----------
    numPoint: int
        number of time points
    numFold: int
        number of pairs of testIndices and trainIndices
    
    Returns:
    --------
    list of pairs of train indices, test indices
    """
    indices = range(numPoint)
    for remainder in range(numFold):
        testIndices = []
        for idx in indices:
            if idx % numFold == remainder:
                testIndices.append(idx)
        trainIndices = list(set(indices).difference(testIndices))
        yield trainIndices, testIndices
        

def plotTS(timeseries, ax=None, linetype="scatter", title="", isPlot=True):
    """
    Plots the variables in a timeseries.
    
    Parameters
    ----------
    timeseries: NamedTimeseries
    ax: Matplotlib.axes
    linetype: str
    title: str
    isPlot: bool
        Show the plot
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = None
    legend = []
    for col in timeseries.colnames:
        legend.append(col)
        if linetype == "scatter":
            _ = ax.scatter(timeseries[TIME], timeseries[col])
        else:
           _ = ax.plot(timeseries[TIME], timeseries[col])
    _ = ax.set_xlabel("time")
    _ = ax.set_title(title)
    _ = plt.legend(legend)
    if not isPlot:
        if fig is not None:
            fig.clear()
    return ax


def makeSyntheticData(fittedTS=FITTED_TS, std=NOISE_STD):
    synthesizer = ObservationSynthesizerRandomErrors(fittedTS=fittedTS, std=std)
    return synthesizer.calculate()
OBSERVED_TS = makeSyntheticData()


##################### CLASSES ###########################
class SimpleFitter(object):

    def __init__(self, model, observedTS, parameterNames,
                lower=LOWER, upper=UPPER, method="leastsq"):
        """
        model: str (Antimony Model)
        observedTS: NamedTimeseries
        paramterNames: list-str
        lower: float
            lower range for parameter value
        upper: float
            upper range for parameter value
        method: str
            optimizer method
        """
        self.rr = te.loada(model)
        self.observedTS = observedTS.copy()
        self.columns = list(self.observedTS.colnames)
        self.allColumns = self.observedTS.allColnames
        self.parameterNames = parameterNames
        self.colnames = self.observedTS.colnames
        self.lower = lower
        self.upper = upper
        self.value = (lower + upper)/2
        self.method = method
        # Internal variables
        self._fittedArr = None
        self._residuals = None
        # Results
        self.params = None
        self.fittedTS = self.observedTS.copy()
        self.residualsTS = None
        
    def mkParams(self):
        """
        Construct lmfit parameters for the parameters.
        """
        parameters = lmfit.Parameters()
        for parameterName in self.parameterNames:
            parameters.add(parameterName,
                          min=self.lower,
                          max=self.upper,
                          value=self.value)
        return parameters
        
    def calcResiduals(self, params):
        """
        Calculate residuals for the fit using the parameters.
        Update self.fittedTS.
        """
        self.rr.reset()  # Put back to time zero
        # Update the simulation parameters
        for name, value in params.valuesdict().items():
            self.rr[name] = value
        fittedArr = self.rr.simulate(0, self.observedTS.end,
                                    len(self.observedTS))
        self._fittedArr = fittedArr.copy()
        fittedArr = fittedArr[:, 1:]  # Delete time column
        observedArr = self.observedTS[self.colnames]
        self._residualsArr = observedArr - fittedArr
        residualsArr = self._residualsArr.flatten()
        return residualsArr
    
    def fit(self, params=None):
        if params is None:
            newParams = self.mkParams()
        else:
            newParams = params.copy()
        # Find the best parameters
        minimizer = lmfit.Minimizer(self.calcResiduals, newParams)
        minimizerResult = minimizer.minimize(method=self.method, max_nfev=100)
        # Record the results
        self.fittedTS = NamedTimeseries(array=self._fittedArr,
              colnames=self.allColumns)
        self.params = minimizerResult.params.copy()
        self.calcResiduals(self.params)  # Update the fitted and residuals
        self.residualsTS = self.observedTS.copy()        
        self.residualsTS[self.columns] = self._residualsArr


class AligningFitter(SimpleFitter):
    """Does fitting with aligning fitted values to timestamps of observed values."""
    
    def __init__(self, model, observedTS, parameterNames, endTime=None,
          numPoint=None, **kwargs):
        """
        model: str (Antimony Model)
        observedTS: NamedTimeseries
        paramterNames: list-str
        endTime: float
            ending time for the simulation
        numPoint: int
            number of points in the simulation
        """
        super().__init__(model, observedTS, parameterNames, **kwargs)
        self.endTime = endTime
        if self.endTime is None:
            self.endTIme = observedTS.end
        self.numPoint = numPoint
        if self.numPoint is None:
            self.numPoint = len(observedTS)
        if self.endTime < observedTS.end:
            msg = "The endTime should be no earlier than observedTS.end"
            raise ValueError(msg)
    
    @staticmethod
    def selectCompatibleIndices(bigTimes, smallTimes):
        """
        Finds the indices such that smallTimes[n] is close to bigTimes[indices[n]]

        Parameters
        ----------
        bigTimes: np.ndarray
        smalltimes: np.ndarray

        Returns
        np.ndarray
        """
        indices = []
        for idx in range(len(smallTimes)):
            distances = (bigTimes - smallTimes[idx])**2
            def getValue(k):
                return distances[k]
            thisIndices = sorted(range(len(distances)), key=getValue)
            index = thisIndices[0]
            if isinstance(index, np.generic):
                index = np.asscalar(index)
            indices.append(index)
        return indices
        
    def calcResiduals(self, params):
        self.rr.reset()  # Put back to time zero
        # Update the simulation parameters
        for name, value in params.valuesdict().items():
            self.rr[name] = value
        fittedArr = self.rr.simulate(0, self.endTime, self.numPoint)
        self._fittedArr = fittedArr.copy()
        indices = AligningFitter.selectCompatibleIndices(fittedArr[:, 0],
              self.observedTS[TIME])
        fittedArr = fittedArr[indices, 1:]  # Delete time column
        observedArr = self.observedTS[self.colnames]
        self._residualsArr = observedArr - fittedArr
        residualsArr = self._residualsArr.flatten()
        return residualsArr
    

class CrossValidator():
    """Performs cross validation using parameter fitting."""
    
    # Dataframe columns and dictionary keys
    PREDICTED = "predicted"
    TRUE = "true"
    FOLD = "fold"
    RSQ = "rsq"
    PARAMETER = "parameter"
    
    def __init__(self, numFold, model, observedTS, parameterNames,
          trueParameterDct=None, **kwargs):
        """
        numFold: int
            number of folds
        model: str (Antimony Model)
        observedTS: NamedTimeseries
        paramterNames: list-str
        trueParameterDct: dict
            key: parameter name, value: parameter value
        kwargs: dict
           optional arguments passed to fitter
        """
        self.numFold = numFold
        self.model = model
        self.observedTS = observedTS.copy()
        self.parameterNames = parameterNames
        self.colnames = self.observedTS.colnames
        self.kwargs = kwargs
        self.trueParameterDct = trueParameterDct
        self.parametersCol = []
        self.rsqs = []

    @staticmethod
    def _calcRsq(observedTS, fittedTS):
        columns = observedTS.colnames
        residualsArr = observedTS[columns] - fittedTS[columns]
        rsq = 1 - np.var(residualsArr)/np.var(observedTS[columns])
        return rsq

    def execute(self):
        numPoint = len(self.observedTS)
        generator = foldGenerator(numPoint, self.numFold)
        for trainIndices, testIndices in generator:
            fitter = AligningFitter(self.model, self.observedTS[trainIndices],
                                  self.parameterNames, endTime=self.observedTS.end,
                                  numPoint=numPoint, **self.kwargs)
            fitter.fit()
            self.parametersCol.append(fitter.params.copy())
            rsq = self._calcRsq(self.observedTS[testIndices],
                  fitter.fittedTS[testIndices])
            self.rsqs.append(rsq)

    def reportParameters(self):
        """
        Constructs a report for the parameter values by fold.
        
        Returns
        -------
        pd.DataFrame
        """
        if self.trueParameterDct is  None:
            raise ValueError("Must construct CrossValidator with trueParameterDct")
        # Construct parameter information
        keys = [CrossValidator.FOLD, CrossValidator.TRUE,
              CrossValidator.PREDICTED, CrossValidator.PARAMETER]
        dct = {}
        for key in keys:
            dct[key] = []
        for fold in range(len(self.parametersCol)):
            for parameterName in self.parameterNames:
                dct[CrossValidator.FOLD].append(fold)
                dct[CrossValidator.PARAMETER].append(parameterName)
                dct[CrossValidator.TRUE].append(
                      self.trueParameterDct[parameterName])
                dct[CrossValidator.PREDICTED].append(
                      self.parametersCol[fold].valuesdict()[parameterName])
        reportDF = pd.DataFrame(dct)
        #
        return reportDF
    
    def reportRsq(self):
        return pd.DataFrame({CrossValidator.RSQ: self.rsqs})
    

if __name__ == '__main__':
    # Tests for foldGenerator
    numFold = 10
    numPoint = 100
    generator = foldGenerator(numPoint, numFold)
    result = [(trainArr, testArr) for trainArr, testArr in generator]
    assert(len(result) == numFold)
    assert(isinstance(n, int) for n in result[0][0])
    # Test for plotTS
    _ = plotTS(FITTED_TS, linetype="line", title="Fitted Data", isPlot=False)
    # Tests for makeSyntheticData
    observedTS = makeSyntheticData(fittedTS=FITTED_TS)
    assert(len(observedTS) == len(FITTED_TS))
    # Tests for SimpleFitter
    fitter = SimpleFitter(MODEL, OBSERVED_TS, PARAMETER_NAMES)
    params = fitter.mkParams()
    k1ParameterValue = params.valuesdict()[PARAMETER_NAMES[0]]
    assert(np.isclose(k1ParameterValue, (LOWER+UPPER)/2))
    #
    dataArr = fitter.calcResiduals(params)
    assert(np.shape(dataArr) == (NUM_POINT*4,))
    #
    fitter.fit()
    fittedResultDct = fitter.params.valuesdict()
    for parameterName, parameterValue in fittedResultDct.items():
        #print(parameterName, parameterValue)
        assert(np.abs(parameterValue - PARAMETER_DCT[parameterName]) < 0.1)
    # Tests for AligningFitter
    size = 50
    observedTS = OBSERVED_TS[list(range(size))]
    fitter = AligningFitter(MODEL, observedTS, PARAMETER_NAMES,
         endTime=OBSERVED_TS.end)
    fitter.fit()
    assert(len(fitter.observedTS) == size)
    assert(len(fitter.params.valuesdict()) == 3)
    # Tests for CrossValidator
    #  _calcRsq
    rsq = CrossValidator._calcRsq(OBSERVED_TS, OBSERVED_TS)
    assert(np.isclose(rsq, 1.0))
    #  execute
    numFold = 5
    validator = CrossValidator(numFold, MODEL, OBSERVED_TS, PARAMETER_NAMES,
                               trueParameterDct=PARAMETER_DCT)
    validator.execute()
    assert(len(validator.rsqs) == numFold)
    assert(len(validator.parametersCol) == numFold)
    assert(isinstance(validator.parametersCol[0], lmfit.Parameters))
    #  reportParameters
    df = validator.reportParameters()
    for key in [CrossValidator.FOLD, CrossValidator.TRUE,
          CrossValidator.PREDICTED, CrossValidator.PARAMETER]:
        assert(key in df.columns)
    assert(len(df) > 0)
    #  reportRsq
    df = validator.reportRsq()
    assert(CrossValidator.RSQ in df.columns)
    assert(len(df) > 0)

    print ("OK!")
