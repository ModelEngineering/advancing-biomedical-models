"""Plot Utilities."""

import common_python.constants as cn
from common_python.util import util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn


# Plot options
XLABEL = "xlabel"
YLABEL = "ylabel"
TITLE = "title"


###### INTERNAL UTILITIES ######
def _getAxis(ax):
  if ax is None:
    ax = plt.gca()
  return ax

def _getValue(key, kwargs):
  if key in kwargs.keys():
    return kwargs[key]
  else:
    return None

def _setValue(key, kwargs, func):
  val = _getValue(key, kwargs)
  if val is not None:
    func(val)


########### EXTERNALLY USED ########################

def plotTrinaryHeatmap(df, ax=None, is_plot=True, **kwargs):
  """
  Plots a heatmap for a dataframe with trinary values: -1, 1, 0
  :param plt.Axis ax:
  :param bool is_plot: shows plot if True
  :param dict kwargs: plot options
  """
  # Setup the data
  df_plot = df.applymap(lambda v: np.nan if np.isclose(v, 0)
      else v)
  # Plot construct
  if ax is None:
    plt.figure(figsize=(16, 10))
  ax = _getAxis(ax)
  columns = df_plot.columns
  ax.set_xticks(np.arange(len(columns)))
  ax.set_xticklabels(columns)
  heatmap = plt.pcolor(df_plot, cmap='jet')
  plt.colorbar(heatmap)
  if XLABEL in kwargs:
    plt.xlabel(kwargs[XLABEL])
  if YLABEL in kwargs:
    plt.ylabel(kwargs[YLABEL])
  if TITLE in kwargs:
    plt.title(kwargs[TITLE])
  if is_plot:
    plt.show()

def plotCategoricalHeatmap(df, is_plot=False, xoffset=0.5, 
    yoffset=0.5, ax=None, **kwargs):
  """
  Plots a heatmap of numerical values with categorical
  x and y axes.
  Row indices are the y-axis; columns are the x-axis
  :param pd.DataFrame df:
  :param int offset: how much labels are offset
  :param dict kwargs: plot options
  :return ax:
  """
  if _getValue(cn.PLT_FIGSIZE, kwargs) is not None:
    plt.figure(figsize=_getValue(PLT_FIGSIZE, kwargs))
  if ax is None:
    ax = plt.gca()
  ax.set_xticks(np.arange(len(df.columns)) + xoffset)
  ax.set_xticklabels(df.columns)
  ax.set_yticks(np.arange(len(df.index)) + yoffset)
  ax.set_yticklabels(df.index)
  cmap = _getValue(cn.PLT_CMAP, kwargs)
  if cmap is None:
    cmap = 'jet'
  if ('vmin' in kwargs) and ('vmax' in kwargs):
    heatmap = plt.pcolor(df, cmap=cmap,
        vmin=kwargs['vmin'], vmax=kwargs['vmax'])
  else:
    heatmap = plt.pcolor(df, cmap=cmap)
  plt.colorbar(heatmap)
  _setValue(cn.PLT_XLABEL, kwargs, plt.xlabel)
  _setValue(cn.PLT_YLABEL, kwargs, plt.ylabel)
  _setValue(cn.PLT_TITLE, kwargs, plt.title)
  if is_plot:
    plt.show()
  return heatmap

def plotCorr(df, is_plot=True, **kwargs):
  """
  Plots correlation of features (columns).
  :param pd.DataFrame df:  rows are instances; columns are features
  :param dict kwargs: plot options
     supported: cmap, xlabel, title, ylabel
  :return seaborn.matrix.ClusterGrid:
  """
  df_corr = df.corr()
  df_corr = df_corr.applymap(lambda v: 0 if np.isnan(v) else v)
  if _getValue(cn.PLT_CMAP, kwargs) is None:
    cmap = "seismic"
  cg = seaborn.clustermap(df_corr, col_cluster=True, vmin=-1, vmax=1,
      cbar_kws={"ticks":[-1, 0, 1]}, cmap=cmap)
  _ = cg.ax_heatmap.set_xticklabels([])
  _ = cg.ax_heatmap.set_xticks([])
  _ = cg.ax_heatmap.set_yticklabels([])
  _ = cg.ax_heatmap.set_yticks([])
  _setValue(cn.PLT_TITLE, kwargs, cg.fig.suptitle)
  _setValue(cn.PLT_XLABEL, kwargs, cg.ax_heatmap.set_xlabel)
  _setValue(cn.PLT_YLABEL, kwargs, cg.ax_heatmap.set_ylabel)
  if is_plot:
    cg.fig.show()
  return cg
