'''Helper Functions for Design of One Factor at a Time Experiments'''

import constants as cn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import fftpack
import tellurium as te
import seaborn as sns
import wolf_model as wm


################## CONSTANTS #############################
AMPLIDX = 1  # Index of amplitude values
FONTSIZE = 16
FREQIDX = 0  # Index of frequency values
TIME = "time"
SMALLEST_PCT = -100  # Smallest percent change in a parameter value
TIME_OFFSET = 150


###################### FUNCTIONS #####################
def arrToDF(arr, isSimulation=True):
    """
    Converts a NamedArray into a DataFrame.
    If it is simulation output, makes TIME the index.

    Parameters
    ----------
    arr: NamedArray
    
    Returns
    -------
    DataFrame
        Removes "[" , "]" from the names of species
    """
    columns = [c[1:-1] if c[0] == "[" else c for c in arr.colnames]
    df = pd.DataFrame(arr, columns=columns)
    if isSimulation:
        df = df.set_index(TIME)
    return df

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
    NamedArray: results of simulation
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
    return arrToDF(data)

def plotOverTime(df, title="", figsize=(8, 4), isPlot=True):
    """
    Plots a simulation dataframe

    Parameters
    ----------
    df: DataFrame (SimulationOutput format)
    title: str
    figsize: (float, float)
        size of figure
    isPlot: bool
        Show the plot
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    arr = df.values
    p = ax.plot(df.index, arr)
    _ = ax.legend(p, df.columns, bbox_to_anchor=(1.05, 1), loc='upper left')
    _ = ax.set_title(title)
    if isPlot:
        plt.show()

def calcFFTPeak(molecule, df, **kwargs):
    """
    Obtains the highest amplitude frequency and value for the molecule.
    
    Parameters
    ----------
    molecule: str
    df: DataFrame
    kwargs: dict
        arguments passed to calcFFT
    
    Returns
    -------
    frequency: float
    amplitude: float
    """
    frequencies, amplitudes = calcFFT(molecule, df, **kwargs)
    # Find the indices of the largest amplitudes
    sortedIndices = sorted(range(len(frequencies)),
          key=lambda i: amplitudes[i], reverse=True)
    topIdx = sortedIndices[0]
    return frequencies[topIdx], amplitudes[topIdx]

def calcFFT(molecule, df, offset=TIME_OFFSET):
    """
    Calculate the FFT for a molecule in the simulation output.
    The calculation does not include amplitudes at a frequency of 0.
    
    Parameters
    ----------
    molecule: str
    df: DataFrame
    offset: int
        Initial data that are not included in the FFT calculation
        
    Returns
    -------
    list-float, list-float
        freqs, fftValues
    """
    # Returns frequencies and abs(fft) for a chemical species (molecule)
    col = molecule
    values = df[col].values
    numPoint = len(df)
    count = numPoint - offset
    endTime = max(df.index)
    startTime= min(df.index)
    span = (endTime - startTime)/numPoint
    freqs = fftpack.fftfreq(count, span)
    fftValues = np.abs(fftpack.fft(values[offset:]))
    # Eliminate frequency of 0
    return freqs[1:], fftValues[1:]

def runFFTExperiment(parameterDct, **kwargs):
    """
    Runs an experiment by changing parameters by the specified
    fractions and calculating FFT peak frequencies and amplitudes.
    
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
    df = runSimulation(parameterDct, **kwargs)
    frequencyDct = {}
    amplitudeDct = {}
    for molecule in df.columns:
        frequency, amplitude = calcFFTPeak(molecule, df)
        frequencyDct[molecule] = frequency
        amplitudeDct[molecule] = amplitude
    return pd.Series(frequencyDct), pd.Series(amplitudeDct)
