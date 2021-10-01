import util_doe1 as doe1
import wolf_model as wm
import constants as cn

import lmfit
import pandas as pd
import numpy as np
import unittest
import matplotlib
matplotlib.use('TkAgg')

############ CONSTANTS #############
IGNORE_TEST = False
IS_PLOT = False
NROWS = 10
NROWS_SUBSET = 5
NCOLS = 3
LENGTH = NROWS*NCOLS
INDICES = range(NROWS)


################### Tests ############
class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.arr= wm.WOLF_DATA
        self.model = wm.WOLF_MODEL
        self.df = doe1.arrToDF(self.arr)

    def testArrToDF(self):
        if IGNORE_TEST:
            return
        df = doe1.arrToDF(self.arr)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(df.index.name, cn.COL_TIME)
        #
        df = doe1.arrToDF(self.arr, isSimulation=False)
        self.assertTrue(cn.COL_TIME in df.columns)

    def testRunSimulation(self):
        if IGNORE_TEST:
            return
        parameterDct = {}
        df = doe1.runSimulation(parameterDct, model=self.model)
        self.assertEqual(df.index.name, cn.COL_TIME)
        parameterDct = {wm.WOLF_PRMS[0]: -10}
        df1 = doe1.runSimulation(parameterDct, model=self.model)
        self.assertFalse(df.equals(df1))
        
    def testPlotOverTime(self):
        if IGNORE_TEST:
            return
        # Smoke test
        doe1.plotOverTime(self.df, figsize=(8, 16), isPlot=IS_PLOT)
        
    def testCalcFFT(self):
        if IGNORE_TEST:
            return
        freqs, fftValues = doe1.calcFFT("Glucose", self.df)
        self.assertGreater(max(fftValues), 90)
        
    def testCalcFFTPeak(self):
        if IGNORE_TEST:
            return
        frequency, amplitude = doe1.calcFFTPeak("Glucose", self.df)
        self.assertTrue(frequency > 5.0)
        self.assertTrue(amplitude > 90)
        
    def testRunFFTExperiment(self):
        if IGNORE_TEST:
            return
        frequencySer, amplitudeSer = doe1.runFFTExperiment({})
        molecule = "Glucose"
        baseGlucoseFrequency, baseGlucoseAmplitude = doe1.calcFFTPeak(
              molecule, self.df)
        self.assertEqual(frequencySer.loc[molecule], baseGlucoseFrequency)
        self.assertEqual(amplitudeSer.loc[molecule], baseGlucoseAmplitude)
    #
    if False:
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

  
    

if __name__ == '__main__':
  unittest.main()
