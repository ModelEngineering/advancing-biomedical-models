'''Tests for modeling game.'''

from common_python.util import util
util.addPath("common_python", 
    sub_dirs=["common_python", "tellurium"])
import model_fitting as mf
import modeling_game as mg

import lmfit
import numpy as np
import pandas as pd

CSV_FILE = "common_python/tests/tellurium/wild.csv"
# Modified model
MODEL1 = '''
// Created by libAntimony v3.9.4
model *pathway()

  // Compartments and Species:
  species INPUT, P1, mRNA1, P2, mRNA2, P3, mRNA3, P4, mRNA4, P5, mRNA5, P6;
  species mRNA6, P7, mRNA7, P8, mRNA8;
  
  J1:  => mRNA1; L1 + Vm1*((K1_1*INPUT^H1 + K2_1*P4^H1 + K1_1*K3_1*INPUT^H1*P4^H1)/(1 + K1_1*INPUT^H1 + K2_1*P4^H1 + K1_1*K3_1*INPUT^H1*P4^H1)) - d_mRNA1*mRNA1;
  F1:  => P1; a_protein1*mRNA1 - d_protein1*P1;
  J2:  => mRNA2; L2 + Vm2*(K1_2*P4^H2/(1 + K1_2*P4^H2)) - d_mRNA2*mRNA2;
  F2:  => P2; a_protein2*mRNA2 - d_protein2*P2;
  J3:  => mRNA3; L3 + Vm3*(K1_3*P6^H3/(1 + K1_3*P6^H3)) - d_mRNA3*mRNA3;
  F3:  => P3; a_protein3*mRNA3 - d_protein3*P3;
  J4:  => mRNA4; L4 + Vm4*(1/(1 + K1_4*P2^H4)) - d_mRNA4*mRNA4;
  F4:  => P4; a_protein4*mRNA4 - d_protein4*P4;
  J5:  => mRNA5; L5 - d_mRNA5*mRNA5;
  F5:  => P5; a_protein5*mRNA5 - d_protein5*P5;
  J6:  => mRNA6; L6 + Vm6*(K1_6*P7^H6/(1 + K1_6*P7^H6 + K2_6*P1^H6 + K1_6*K2_6*P7^H6*P1^H6)) - d_mRNA6*mRNA6;
  F6:  => P6; a_protein6*mRNA6 - d_protein6*P6;
  J7:  => mRNA7; L7 +  Vm7*( K1_7*P1^H7/(1 + K1_7*P1^H7) + 1/(1 + K2_7*P7^H7)) - d_mRNA7*mRNA7;
  F7:  => P7; a_protein7*mRNA7  - d_protein7*P7;
  J8:  => mRNA8; L8 + Vm8*(1/(1 + K1_8*P1^H8)) - d_mRNA8*mRNA8;
  F8:  => P8; a_protein8*mRNA8 - d_protein8*P8;
  '''

MODEL2 = '''

 // Constants to estimate
 Vm7 = 0;
 K1_7 = 0;
 K2_7 = 0;
 H7 = 0;
 K8_7 = 0;
 K7_8 = 0;

 // Species initializations:
  INPUT = 1;
  P1 = 0;
  mRNA1 = 0;
  P2 = 0;
  mRNA2 = 0;
  P3 = 0;
  mRNA3 = 0;
  P4 = 0;
  mRNA4 = 0;
  P5 = 0;
  mRNA5 = 0;
  P6 = 0;
  mRNA6 = 0;
  P7 = 0;
  mRNA7 = 0;
  P8 = 0;
  mRNA8 = 0;

  // Variable initializations:
  d_protein1 = 0.01576525;
  d_mRNA1 = 0.600013;
  L1 = 0.0284569399501349;
  Vm1 = 1.16380673480284;
  a_protein1 = 0.089105566248939;
  H1 = 4.52340391321994;
  K1_1 = 0.0269204907071558;
  K2_1 = 0.0169635567504703;
  K3_1 = 0.0114278645720656;
  d_protein2 = 0.0100753359178861;
  d_mRNA2 = 0.607263145327485;
  
  L2 = 0.01126375;
  Vm2 = 0.855433454948057;
  a_protein2 = 0.0825104648147814;
  H2 = 3.21939257313515;
  K1_2 = 0.0170170903653747;
  
  d_protein3 = 0.0165270958726424;
  d_mRNA3 = 1.4096553751623;
  L3 = 0.0108824170117358;
  Vm3 = 1.61383118795785;
  a_protein3 = 0.118672807163739;
  H3 = 4.57189341195625;
  K1_3 = 0.0133069236136431;
  
  d_protein4 = 0.0205716618573404;
  d_mRNA4 = 1.19069657063437;
  L4 = 0.0142166568749038;
  Vm4 = 0.916251896011744;
  a_protein4 = 0.0862964088164644;
  H4 = 5.00512303222327;
  K1_4 = 0.0179894288457716;
  
  d_protein5 = 0.0180685727313577;
  d_mRNA5 = 0.911653907722886;
  L5 = 0.0124824037745626;
  a_protein5 = 0.106293056264931;
  d_protein6 = 0.0178004316181647;
  
  d_mRNA6 = 0.67287496171115;
  L6 = 0.0166516087433626;
  Vm6 = 0.889302076744445;
  a_protein6 = 0.0890528700251159;
  H6 = 5.58112408673455;
  K1_6 = 0.0139445776013774;
  K2_6 = 0.0121764364668572;
  
  d_protein7 = 0.0206180615545929;
  d_mRNA7 = 0.618711430584466;
  L7 = 0.0178636133026099;
  a_protein7 = 0.0764169841455256;
  
  d_protein8 = 0.0131749080364666;
  d_mRNA8 = 1.17483328491068;
  L8 = 0.0165535374304033;
  Vm8 = 0.87881445337468;
  a_protein8 = 0.103749989801903;
  H8 = 2.17775388441324;
  K1_8 = 0.0168599518440462;

  // Other declarations:
  const d_protein1, d_mRNA1, L1, Vm1, a_protein1, H1, K1_1, K2_1, K3_1, d_protein2;
  const d_mRNA2, L2, Vm2, a_protein2, H2, K1_2, d_protein3, d_mRNA3;
  const L3, Vm3, a_protein3, H3, K1_3, d_protein4, d_mRNA4, L4;
  const Vm4, a_protein4, H4, K1_4, d_protein5, d_mRNA5, L5;
  const a_protein5, d_protein6, d_mRNA6, L6, Vm6, a_protein6;
  const H6, K1_6, K2_6, d_protein7, d_mRNA7, L7, a_protein7;
  const d_protein8, d_mRNA8, L8, Vm8, a_protein8, H8, K1_8;
end
'''
MODEL = MODEL1 + MODEL2

def testMakeParameters():
  def test(constants, values=None):
    parameters = mg.makeParameters(constants, values=values)
    assert(len(constants) == len(parameters.valuesdict().keys()))
    if values is not None:
      trues = [v == parameters.valuesdict()[c] for
          c, v in zip(constants, values)]
      assert(all(trues))
  #
  constants = ["Vm1", "d_protein2"]
  test(constants) 
  test(constants, values=[0.5, 0.01]) 

def testDoBootstrap3():
  constants = ["Vm"]
  sim_time = 300
  num_points = int(sim_time/10)
  parameters = mg.makeParameters(constants)
  df_rnaseq = pd.read_csv(CSV_FILE)
  df_obs = df_rnaseq.loc[df_rnaseq.index[range(num_points)], :]
  statistic_dict = mf.doBootstrap(df_obs, MODEL, parameters,
                                             num_points=num_points, 
                                             method=mf.ME_BOTH,
                                             sim_time=sim_time)

def _generateData():
  result = mf.runSimulation(model=MODEL, sim_time=1200,
      num_points=120)
  return result.data

def testPlotSimulation():
  # Only smoke tests
  data = _generateData()
  mg.plotSimulation(data, MODEL, is_plot_model=False,
      title="title",
      is_plot_observations=False)
  mg.plotSimulation(data, MODEL)

def testRunExperiment():
  # Only smoke tests
  df_data = mf.cleanColumns(pd.read_csv("wild.csv"))
  parameters = mg.runExperiment(df_data, MODEL, parameters=["Vm1"])
  assert(len(parameters.valuesdict().keys()) == 1)
  # Smoke test for options
  parameters = mg.runExperiment(df_data, MODEL, parameters=["Vm1"],
      title="title",
      is_plot_observations=False)
 
 
if __name__ == '__main__':
  testPlotSimulation()
  print("OK.")
