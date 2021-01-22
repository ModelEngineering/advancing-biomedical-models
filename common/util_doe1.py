'''Helper Functions for Design of One Factor at a Time Experiments'''


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import fftpack
import tellurium as te
import seaborn as sns
import wolf_model as wm
from SBstoat.namedTimeseries import NamedTimeseries, TIME


################## CONSTANTS #############################
AMPLIDX = 1  # Index of amplitude values
FONTSIZE = 16
FREQIDX = 0  # Index of frequency values
TIME = "time"
SMALLEST_PCT = -100  # Smallest percent change in a parameter value
TIME_OFFSET = 100


###################### FUNCTIONS #####################
def runSimulation(parameterDct, roadrunner=None, model=wm.WOLF_MODEL,
      startTime=wm.START, endTime=wm.END, numPoint=wm.NUMPT):
    """
    Runs a simulation for parameters with percentage changes from baseline values.

    Parameters
    ----------
    parameterDct: dict
       key: parameter
       value: float
           percent change
    roadrunner: ExtendedRoadRunner
    startTime: float
    endTime: float
    numPoint: int
       
    Returns
    -------
    namedArray: results of simulation
    """
    if roadrunner is None:
        roadrunner = te.loada(model)
    else:
        roadrunner.resetAll()
    for parameter, percent in parameterDct.items():
        baseValue = roadrunner[parameter]
        roadrunner[parameter] = baseValue*(1 + 0.01*percent)
    # Run the simulation
    data = roadrunner.simulate(startTime, endTime, numPoint)
    return data

def calculateFft(molecule, data, offset=100):
    """
    Calculate the FFT for a molecule in the simulation output.
    The calculation does not include amplitudes at a frequency of 0.
    
    Parameters
    ----------
    molecule: str
    data: NamedArray
    offset: int
        Initial data that are not included in the FFT calculation
             
    Returns
    -------
    list-float, list-float
        freqs, fftValues
    """
    # Returns frequencies and abs(fft) for a chemical species (molecule)
    if molecule in data.colnames:
        col = molecule
    else:
        col = "[%s]" % molecule
    values = data[col]
    numPoint = len(data)
    count = numPoint - offset
    endTime = data["time"][-1]
    freqs = fftpack.fftfreq(count, endTime/numPoint)
    fftValues = np.abs(fftpack.fft(values[offset:]))
    # Eliminate frequency of 0
    freqs = freqs[1:]
    fftValues = fftValues[1:]
    return freqs, fftValues

def getFrequencyAmplitude(molecule, data, **kwargs):
    """
    Obtains the highest amplitude frequency and value for the molecule.
    
    Parameters
    ----------
    molecule: str
    data: NamedArray/Namedtimeseries
    kwargs: dict
        arguments passed to calculateFft
    
    Returns
    -------
    frequency: float
    amplitude: float
    """
    # Return True if the expected frequency is among the topN frequencies with the largest amplitudes
    frequencies, amplitudes = calculateFft(molecule, data=data, **kwargs)
    # Find the indices of the largest amplitudes
    sortedIndices = sorted(range(len(frequencies)), key=lambda i: amplitudes[i], reverse=True)
    topIdx = sortedIndices[0]
    return frequencies[topIdx], amplitudes[topIdx]

def calculateFft(molecule, data, offset=TIME_OFFSET):
    """
    Calculate the FFT for a molecule in the simulation output.
    The calculation does not include amplitudes at a frequency of 0.
    
    Parameters
    ----------
    molecule: str
    data: NamedArray/Namedtimeseries
    offset: int
        Initial data that are not included in the FFT calculation
        
    Returns
    -------
    list-float, list-float
        freqs, fftValues
    """
    # Returns frequencies and abs(fft) for a chemical species (molecule)
    if molecule in data.colnames:
        col = molecule
    else:
        col = "[%s]" % molecule
    values = data[col]
    numPoint = len(data)
    count = numPoint - offset
    endTime = data[TIME][-1]
    startTime= data[TIME][0]
    span = (endTime - startTime)/numPoint
    freqs = fftpack.fftfreq(count, span)
    fftValues = np.abs(fftpack.fft(values[offset:]))
    # Eliminate frequency of 0
    freqs = freqs[1:]
    fftValues = fftValues[1:]
    return freqs, fftValues

def runExperiment(parameterDct, **kwargs):
    """
    Runs an experiment by changing parameters by the specified fractions and calculating responses.
    
    Parameters
    ----------
    parameterDct: dict
       key: parameter name
       value: percent change the parameter
    kwargs: dict
       keyword arguments passed to runSimulation
       
    Returns
    -------
    frequencySer: pd.Series
        index: molecule
        value: frequency with largest amplitude
    amplitudeSer: pd.Series
        index: molecule
        value: largest amplitude
    """
    data = runSimulation(parameterDct, **kwargs)
    frequencyDct = {}
    amplitudeDct = {}
    molecules = [s[1:-1] for s in data.colnames if s != TIME]
    for molecule in molecules:
        frequency, amplitude = getFrequencyAmplitude(molecule, data)
        frequencyDct[molecule] = frequency
        amplitudeDct[molecule] = amplitude
    return pd.Series(frequencyDct), pd.Series(amplitudeDct)

def runExperiments(parameter, percents, isRelative=True):
    """
    Runs experiments for one parameter of the model at different percent changes in the parameter value (levels).
    
    Parameter
    ---------
    parameter: str
    percents: list-float
        percent change in parameter
    isRelative: bool
        True: values are percent changes relative to baseline
        False: absolute value
        
    Returns
    -------
    frequencyDF: DataFrame
        index: percent
        columns: molecule
        values: percent change in frequency w.r.t. baseline
    amplitudeDF: DataFrame
        index: percent
        columns: molecule
        values: depends on isRelative
    """
    # Calculate the baseline values
    baseFrequencySer, baseAmplitudeSer = runExperiment({})
    #
    def calcResponseSer(ser, isFrequency=True):
        """
        Calculates the relative response.
        
        Parameters
        ----------
        ser: pd.Series
            index: molecule
            value: absolute respoinse
        isFrequency: bool
            if True, frequency response; else, amplitude response
            
        Returns
        -------
        pd.Series
        """
        if not isRelative:
            return ser
        if isFrequency:
            baseSer = baseFrequencySer
        else:
            baseSer = baseAmplitudeSer
        resultSer = 100*(ser - baseSer)/baseSer
        return resultSer
    #
    def calcLevelDF(isFrequency=None):
        """
        Calculates the dataframe of levels dataframe.
        
        Parameter
        --------
        isFrequency: bool
            If True, frequency response. Otherwise, amplitude response
            
        Returns
        -------
        pd.DataFrame
            index: tuple-int
                levels of parameters
            columns: str
                molecule
            values: response
        """
        if isFrequency is None:
            raise ValueError("Must specify isFrequency!")
        sers = []  # Collection of experiment results
        for percent in percents:
            parameterDct = {parameter: percent}
            frequencySer, amplitudeSer = runExperiment(parameterDct)
            if isFrequency:
                ser = frequencySer
            else:
                ser = amplitudeSer
            adjSer = calcResponseSer(ser, isFrequency=isFrequency)
            sers.append(pd.DataFrame(adjSer).transpose())
        resultDF = pd.concat(sers)
        resultDF.index = percents
        return resultDF
    #
    frequencyDF = calcLevelDF(isFrequency=True)
    amplitudeDF = calcLevelDF(isFrequency=False)
    return frequencyDF, amplitudeDF


if __name__ == '__main__':
    freqs, fftValues = calculateFft("Glucose", wm.WOLF_DATA)
    assert(max(fftValues) > 90)  # Top frequency should have a large magnitude
    #
    frequency, amplitude = getFrequencyAmplitude("Glucose", wm.WOLF_DATA)
    assert(frequency > 5.0)
    assert(amplitude > 90)
    #
    freqs, fftValues = calculateFft("Glucose", wm.WOLF_DATA)
    assert(max(fftValues) > 90)  # Top frequency should have a large magnitude
    #
    frequencySER, amplitudeSER = runExperiment({"J1_Ki": 0.03})
    assert(len(frequencySER) == len(amplitudeSER))
    #
    percents = [-7, 0, 7]
    fDF, aDF = runExperiments("J1_Ki", percents)
    assert(np.isclose(fDF.loc[percents[0], "Glucose"], -1*fDF.loc[percents[-1], "Glucose"]) )
    assert(aDF.loc[percents[-1], "Glucose"] < 0 )
    assert(aDF.loc[percents[0], "Glucose"] > 0 )
    print ("OK!")
