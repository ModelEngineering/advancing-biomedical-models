from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])

from common_python.tellurium import constants as cn
from common_python.tellurium import model_fitting as mf
from common_python.tellurium import modeling_game as mg
from common_python.tellurium import gene_network as gn
from common_python.tellurium import gene_analyzer as ga
from common_python.tellurium.gene_network import  \
    GeneDescriptor, GeneReaction, GeneNetwork
from common_python.testing import helpers

import lmfit
import pandas as pd
import numpy as np
import unittest

IGNORE_TEST = False
DESC_STG = "7-7"
START_TIME = 0
END_TIME = 300


###########################################################
class TestGeneAnalyzer(unittest.TestCase):

  def setUp(self):
    if IGNORE_TEST:
      return
    self._init()

  def _init(self):
    self.analyzer = ga.GeneAnalyzer()
    self.analyzer._initializeODScope(DESC_STG, START_TIME, END_TIME)
    self.analyzer._initializeODPScope(
        self.analyzer.network.new_parameters)

  def testConstructor(self):
    if IGNORE_TEST:
      return
    self._init()
    self.assertTrue(
        isinstance(self.analyzer.parameters, lmfit.Parameters))
    self.assertTrue("P1" in self.analyzer.namespace.keys())
    self.assertTrue(helpers.isValidDataFrame(self.analyzer._df_mrna,
        self.analyzer._df_mrna.columns))

  def testMakePythonExpression(self):
    if IGNORE_TEST:
      return
    self._init()
    self.analyzer._initializeODScope(DESC_STG, START_TIME, END_TIME)
    result = ga.GeneAnalyzer._makePythonExpression(
        self.analyzer.reaction.mrna_kinetics)
    keys = self.analyzer.network.new_parameters.valuesdict().keys()
    for key in keys:
        self.analyzer.namespace[key] = 0
    self.assertTrue(isinstance(eval(result, self.analyzer.namespace),
        float))

  def testCalcKinetics(self):
    if IGNORE_TEST:
      return
    self._init()
    def test(time):
      y_arr = [0]
      result = ga.GeneAnalyzer._calcKinetics(y_arr,
          time, self.analyzer)
      trues = [x >= 0 for x in result]
      self.assertTrue(all(trues))
      self.assertGreater(result[0], 0)
    #
    test(0)
    test(0.012)

  def testCalcMrnaEstimates1(self):
    if IGNORE_TEST:
      return
    self._init()
    self.assertTrue(self.analyzer.arr_est is None)
    self.analyzer._calcMrnaEstimates(
        self.analyzer.network.new_parameters)
    self.assertTrue(isinstance(self.analyzer.arr_est, np.ndarray))
    self.assertEqual(len(self.analyzer.arr_est),
        int(END_TIME/ga.TIME_UNIT))

  def testCalcMrnaEstimates2(self):
    if IGNORE_TEST:
      return
    self._init()
    parameters = mg.makeParameters(
        ['Vm7', 'H7', 'K1_7'],
        values=[0.01000148, 1.52295321, 2.17466603]
        )
    self.analyzer._calcMrnaEstimates(parameters)
    trues = [v > 0.3 for v in self.analyzer.arr_est[1:]]
    self.assertTrue(all(trues))

  def testDo(self):
    if IGNORE_TEST:
      return
    analyzer2 = ga.GeneAnalyzer()
    analyzer2.do(DESC_STG, start_time=END_TIME/2,
        end_time=END_TIME, min_rsq=1.0,
        max_iteration=5)
    #
    analyzer1 = ga.GeneAnalyzer()
    analyzer1.do(DESC_STG, end_time=END_TIME, min_rsq=1.0,
        max_iteration=5)
    self.assertTrue(isinstance(analyzer1.rsq, float))
    #
    self.assertGreater(len(analyzer1.arr_est), len(analyzer2.arr_est))

  def testDo1(self):
    if IGNORE_TEST:
      return
    analyzer = ga.GeneAnalyzer()
    analyzer.do("7-7", end_time=300, max_iteration=10, min_rsq=0.8)
    self.assertGreater(analyzer.rsq, 0.3)

  def testProteinInitializations(self):
    if IGNORE_TEST:
      return
    df_mrna, compileds = ga.GeneAnalyzer.proteinInitializations(
        cn.MRNA_PATH)
    self.assertTrue("code" in str(compileds[0].__class__))
    self.assertTrue(helpers.isValidDataFrame(df_mrna,
        df_mrna.columns))

  def testProteinDydt(self):
    if IGNORE_TEST:
      return
    MAX = 10
    times = [10.0*n for n in range(MAX)]
    df_mrna, compileds = ga.GeneAnalyzer.proteinInitializations(
        cn.MRNA_PATH)
    y0_arr = np.repeat(0, gn.NUM_GENE + 1)
    y_arr = np.array(y0_arr)
    y_arrs = []
    for time in times:
      y_arr = ga.GeneAnalyzer._proteinDydt(y_arr, time,
          df_mrna, compileds)
      y_arrs.append(y_arr)
    trues = [np.isclose(v, 0) for v in y_arrs[0]]
    self.assertTrue(all(trues))
    trues = [v > 0.0 for v in y_arrs[-1]]
    self.assertTrue(all(trues))

  def testMakeProtein(self):
    if IGNORE_TEST:
      return
    df = ga.GeneAnalyzer.makeProteinDF(end_time=30, is_write=False)
    columns = [gn.GeneReaction.makeProtein(n)
        for n in range(1, gn.NUM_GENE+1)]
    columns.insert(0, cn.TIME)
    self.assertTrue(helpers.isValidDataFrame(df, columns))

  def testPlot(self):
    if IGNORE_TEST:
      return
    analyzer = ga.GeneAnalyzer()
    analyzer.do("7", end_time=100)
    analyzer.plot()

  def testMakeParameterDF(self):
    if IGNORE_TEST:
      return
    analyzer = ga.GeneAnalyzer()
    analyzer.do("7-8", end_time=100)
    df = analyzer.makeParameterDF()
    expecteds = set(['Vm7', 'K1_7', 'H7'])
    self.assertEqual(len(expecteds.symmetric_difference(
        analyzer.parameters.valuesdict().keys())), 0)
    

if __name__ == '__main__':
  unittest.main()
