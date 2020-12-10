from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])

import common_python.tellurium.constants as cn
from common_python.tellurium import modeling_game as mg
from common_python.tellurium import gene_network as gn
from common_python.tellurium import run_game as rg
from common_python.util import persister

import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
PARAM_PATH = "/tmp/parameters.csv"
MODEL_PATH = "/tmp/full_model.txt"
#PERSISTER_PATH = "/tmp/test_run_game.pcl"
#PERSISTER = persister.Persister(PERSISTER_PATH)
#if not PERSISTER.isExist():
#  ANALYZERS = []
#  ANALYZERS.append(rg.evaluate(["1+4"]))
#  ANALYZERS.append(rg.evaluate(["7+1"]))
#  PERSISTER.set(ANALYZERS)
#ANALYZERS = PERSISTER.get()
ANALYZERS = []
ANALYZERS.append(rg.evaluate(["7"]))
ANALYZERS.append(rg.evaluate(["3"]))

###########################################################
class TestFunctions(unittest.TestCase):

  def _init(self):
    if IGNORE_TEST:
      return
    _ = rg.saveAnalysisResults(ANALYZERS,
        parameters_path=PARAM_PATH, model_path=MODEL_PATH)

  def testEvaluate(self):
    # Smoke test
    desc_stg = "1+4"
    analyzer = rg.evaluate([desc_stg])
    self.assertEqual(desc_stg, str(analyzer.descriptor))

  def testAnalysisResults(self):
    df_params, model = rg.saveAnalysisResults(ANALYZERS)
    self.assertTrue(isinstance(df_params, pd.DataFrame))
    self.assertTrue(isinstance(model, str))

  def testRunModel(self):
    # Smoke test
    self._init()
    rg.runModel(parameters_path=PARAM_PATH, model_path=MODEL_PATH)
    

if __name__ == '__main__':
  unittest.main()
