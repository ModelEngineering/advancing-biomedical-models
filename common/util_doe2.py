"""Helper Functions for Design of Experiments With Two Factors in Combination."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import tellurium as te
import seaborn as sns
import util_doe1
import wolf_model as wm


def runExperimentsTwoParameters(parameter1, parameter2, percent1s, percent2s,
      isRelative=True):
    """
    Runs experiments for changes in multipleparameters of the model at
    different percent changes in the parameter value (levels).
    
    Parameter
    ---------
    parameter1: str
    parameter2: str
    percent1s: list-int
    percent2s: list-int
    isRelative: bool
        True: values are percent changes relative to baseline
        False: absolute value
        
    Returns
    -------
    frequencyDF: DataFrame, amplitude: DataFrame
       index: tuple of percent changes in parameter values
       columns: molecule
       value: depends on isRelative
       index.name: string of tuple (rowParameter, columnParameter)
    """
    INDEX1 = "index1"
    INDEX2 = "index2"
    # Calculate the baseline values
    baseFrequencySer, baseAmplitudeSer = util_doe1.runExperiment({})
    #
    def calc(ser, isFrequency=True):
        if not isRelative:
            return ser
        if isFrequency:
            baseSer = baseFrequencySer
        else:
            baseSer = baseAmplitudeSer
        resultSer = 100*(ser - baseSer)/baseSer
        return resultSer
    #
    def iterateLevels(isFrequency=None):
        if isFrequency is None:
            raise ValueError("Must specify isFrequency!")
        sers = []  # Collection of experiment results
        index1 = []
        index2 = []
        for percent1 in percent1s:
            for percent2 in percent2s:
                #indices.append("%d_%d" % (percent1, percent2))
                index1.append(percent1)
                index2.append(percent2)
                parameterDct = {parameter1: percent1, parameter2: percent2}
                frequencySer, amplitudeSer = util_doe1.runExperiment(parameterDct)
                if isFrequency:
                    ser = frequencySer
                else:
                    ser = amplitudeSer
                adjSer = calc(ser, isFrequency=isFrequency)
                sers.append(pd.DataFrame(adjSer).transpose())
        resultDF = pd.concat(sers)
        resultDF[INDEX1] = index1
        resultDF[INDEX2] = index2
        resultDF = resultDF.set_index([index1, index2])
        for column in [INDEX1, INDEX2]:
            del resultDF[column]
        resultDF.index.name = str((parameter1, parameter2))
        return resultDF
    #
    frequencyDF = iterateLevels(isFrequency=True)
    amplitudeDF = iterateLevels(isFrequency=False)
    return frequencyDF, amplitudeDF

def pivotResponse(responseDF, molecule):
    """
    Creates a DataFrame in which values of the parameters are rows and columns.
    
    Parameters
    ----------
    responseDF: DataFrame
        columns: molecule names
        rows: tuple of percent changes in parameters
        values: response
        index.name: string of tuple (rowParameter, columnParameter)
    molecule: str
        
    Returns
    -------
    DataFrame
        columns: values of columnParameter
        rows: values of rowParameter
        cells: response for molecule
    """
    LEVEL0 = "level_0"
    LEVEL1 = "level_1"
    df = responseDF.reset_index()
    resultDF = pd.pivot_table(df, values=molecule, index=LEVEL0, columns=LEVEL1)
    parameters = eval(responseDF.index.name)
    resultDF.index.name = parameters[0]
    resultDF.columns.name = parameters[1]
    return resultDF

def makeHeatmap(responseDF, molecule, isColorbar=True, ax=None,
      vmin=-100, vmax=100, cbar_ax=None):
    """
    Creates a heatmap showing the interactions of response values for
    two parameters.
    
    Parameters
    ----------
    reponseDF: DataFrame
        index: tuples of percent changes in parameter values
        columns: molecule
        values: response
    molecule: str
    isColorbar: bool
        show the colorbar
    vmin: float
        minimum value in color bar
    vmax: float
        maximum value in color bar
    cbar_ax: Axes
    """
    df = pivotResponse(responseDF, molecule)
    df = df.sort_index(ascending=False)  # Show large values on top
    if ax is None:
        ax = sns.heatmap(df, cmap='seismic', vmin=vmin, vmax=vmax,
              cbar=isColorbar, cbar_ax=cbar_ax)
    else:
        _ = sns.heatmap(df, cmap='seismic', vmin=vmin, vmax=vmax,
              cbar=isColorbar, ax=ax, cbar_ax=cbar_ax)
    return ax

def runStudyTFC(molecule, parameters, percents, isRelative=True,
      isFrequency=True, func=None, figsize=(20, 10)):
    """
    Creates an upper triangular plot of the interactions between 
    parameter pairs in combinations.
    
    Parameters
    ----------
    molecule: str
    parameters: list-str
    percents: list-int
    isRelative: bool
       Results are percent change w.r.t. base
    isFrequency: bool
       Results are for frequencies
    func: function
       Function with the same signature (inputs and outputs)
       as runExperimentsTwoParameters
    figisze: tuple-int
       Size of figures
    """
    if func is None:
        func = runExperimentsTwoParameters
    numParameter = len(parameters)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(numParameter, numParameter)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for rowIdx in range(numParameter):
        parmRowidx = rowIdx
        for colIdx in range(numParameter-rowIdx-1):
            parmColidx = numParameter - colIdx - 1
            frequencyDF, amplitudeDF = func(
                parameters[parmRowidx], parameters[parmColidx], 
                percents, percents, isRelative=isRelative)
            if isFrequency:
                responseDF = frequencyDF
            else:
                responseDF = amplitudeDF
            ax = plt.subplot(gs[rowIdx, colIdx])
            # Label the parameters for each column
            if rowIdx == 0:
                ax.text(1, -0.4, parameters[parmColidx],
                      fontsize=util_doe1.FONTSIZE)
            # Only display colorbar for the last plot
            if (rowIdx == numParameter - 2):
                _ = makeHeatmap(responseDF, molecule, ax=ax,
                      isColorbar=True, cbar_ax=cbar_ax)
            else:
                _ = makeHeatmap(responseDF, molecule, ax=ax, isColorbar=False)
            ax.set_xlabel("")
            # Only display ylabel for left-most plot
            if colIdx == 0:
                ax.set_ylabel(parameters[parmRowidx], fontsize=util_doe1.FONTSIZE)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            # Only show x-tics for bottom-most plot
            if colIdx != (numParameter - rowIdx - 2):
                ax.set_xticklabels([])
    if isFrequency:
        responseType = "Frequency"
    else:
        responseType = "Amplitude"
    if isRelative:
        metricType = "relative"
    else:
        metricType = "absolute"
    suptitle = "%s: %s response (%s)" % (molecule, responseType, metricType)
    plt.suptitle(suptitle, fontsize=2*util_doe1.FONTSIZE)


if __name__ == '__main__':
    percents = [-7, -5, -2, 0, 2, 5, 7]
    TEST_FDF, TEST_ADF = runExperimentsTwoParameters("J1_Ki", "J1_k1",
          percents, percents)
    assert(TEST_ADF.loc[(percents[-1], percents[-1]), "Glucose"] < 0 )
    assert(TEST_FDF.loc[(percents[0], percents[0]), "Glucose"] < 0 )
    #
    df = pivotResponse(TEST_FDF, "Glucose")
    #pd.pivot_table(df, values="Glucose", index="level_0", columns="level_1")
    assert(len(df.index) == len(df.columns))
    #
    _ = makeHeatmap(TEST_FDF, "ATP")
    #
    runStudyTFC("ATP", wm.WOLF_PRMS[0:3], [-5, 0, 5],
          func=runExperimentsTwoParameters,
          isRelative=True, isFrequency=True)    
