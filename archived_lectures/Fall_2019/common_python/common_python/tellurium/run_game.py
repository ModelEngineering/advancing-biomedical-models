"""Top level codes for saving and running simulation models."""

import lmfit
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import constants as cn
import gene_network as gn
import modeling_game as mg
import model_fitting as mf
import gene_analyzer as ga
import util


SIM_TIME = 1200
NUM_POINTS = 120

# Files
MODEL_PATH = os.path.join(cn.TELLURIUM_DIR, "full_model.txt")
PARAM_PATH = os.path.join(cn.TELLURIUM_DIR, "parameters.csv")


def evaluate(desc_stgs, **kwargs):
  """
  Analyzes the quality of a gene description (configuration).
  :param list-str desc_stgs: list of descriptor strings
  :param dict kwargs: keyword arguments passed to GeneAnalyzer.do
  :return GeneAnalyzer:
  """
  analyzer = ga.GeneAnalyzer()
  for desc_stg in desc_stgs:
    analyzer.do(desc_stg, **kwargs)
    title = "%s: Rsq = %1.2f" % (desc_stg, analyzer.rsq)
    plt.figure()
    analyzer.plot(title=title)
  return analyzer

def saveAnalysisResults(analyzers, parameters_path=PARAM_PATH,
                        model_path=MODEL_PATH):
  """
  Saves the parameters and model from the accumulated files.
  :param list-GeneAnalyzer analyzers:
  :param str parameters_path: Write parameters
  :param str model_path: Write model
  """
  # Accumulate descriptors and parameters
  desc_stgs = []
  dfs_params = []
  for analyzer in analyzers:
      dfs_params.append(analyzer.makeParameterDF())
      desc_stgs.append(str(analyzer.descriptor))
  # Write the parameters
  df_param = pd.concat(dfs_params)
  df_param = df_param.set_index(cn.NAME)
  try:
    df_param = df_param.drop(["Vm1"])
  except KeyError:
    pass
  df_param = df_param.reset_index()
  df_param = df_param.drop_duplicates()
  df_param.to_csv(parameters_path)
  # Save the parameters required
  param_dict = {cn.NAME: [], cn.VALUE: []}
  for parameters in analyzers:
    dfs_params.append(analyzer.makeParameterDF())
  # Write the simulation file
  network = gn.GeneNetwork()
  network.update(desc_stgs)
  # Add the new parameters
  parameters = mg.makeParameters(df_param[cn.NAME],
      values=df_param[cn.VALUE])
  [network.addInitialization(a.parameters) for a in analyzers]
  network.generate()
  with open(model_path, "w") as fd:
    fd.write(network.model)
  return df_param, network.model

def runModel(model_path=MODEL_PATH, parameters_path=PARAM_PATH,
    mrna_path=cn.MRNA_PATH):
  """
  Evaluates a previously saved simulation model_path.
  :param str model_path: path to the Tellurium simulation
  :param str parameters_path: path to the parameter file
  """
  # Initializations
  model = getModel(model_path=model_path)
  df_params = pd.read_csv(parameters_path)
  parameters = mg.makeParameters(df_params[cn.NAME],
      df_params[cn.VALUE])
  # Run simulation
  df_mrna = pd.read_csv(mrna_path)
  df_mrna = df_mrna.set_index(cn.TIME)
  df_mrna = df_mrna.drop(df_mrna.index[-1])
  fitted_parameters = mf.fit(df_mrna, model=model,
       parameters=parameters,
      sim_time=SIM_TIME, num_points=NUM_POINTS)
  # Simulate with fitted parameters and plot
  mg.plotSimulation(df_mrna, model, parameters=fitted_parameters,
      is_plot_observations=True, is_plot_model=True,
      is_plot_residuals=True, title=None)

def getModel(model_path=MODEL_PATH):
  """
  :return str: model in the model file
  """
  # Initializations
  with open(MODEL_PATH, "r") as fd:
    model = fd.readlines()
  return "".join(model)
