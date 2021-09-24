from common import util_doe1
from common import wolf_model as wm

import lmfit
import pandas as pd
import numpy as np
import unittest

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
        pass

    def testCalculateFFT(self):
        freqs, fftValues = util_doe1.calculateFft("Glucose", wm.WOLF_DATA)
        self.assertGreater(max(fftValues), 90)
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
