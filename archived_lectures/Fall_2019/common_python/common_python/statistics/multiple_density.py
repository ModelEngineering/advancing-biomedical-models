"""Multiple densities with the same variates."""


import common_python.constants as cn
from common_python.plots import util_plots
from common_python.statistics import density

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


INDEX_MULT = 1000
THIS = "this"
OTHER = "other"
COLORS = ["red", "green", "blue", "grey", "brown"]


class MultipleDensity(object):
  
  def __init__(self, df, variates):
    """
    Determines the variates present in the cells of the DataFrame.
    :param pd.DataFrame df: data matrix
    :param list-object variates: variate variates
    :param return: a density dataframe; columns are features;
        row index are variates; variates are probabilities ([0,1])
    """
    sers = []
    for col in df.columns:
      a_density = density.Density(df[col], variates=variates)
      sers.append(a_density.ser_density)
    self.df = pd.concat(sers, axis=1)
    self.df.columns = df.columns

  def calcSortIndex(self, sort_order=None):
    """
    Creates a sort index for each feature based on the combination
    of probability values for each variate.
    Since values are assumed to be in the range [0, 1], the sort index
    consists of successive groups of 3 numerials (int(100*value)).
    :param list-object sort_order:
    :return pd.Series: index is feature, value is sort_index
    """
    if sort_order is None:
      sort_order = self.df.index.tolist()
    else:
      sort_order = list(sort_order)
      sort_order.sort()
    # Calculate the sort value
    sort_values = {}
    for col in self.df.columns:
      sort_value = 0
      for variate in sort_order:
        variate_index = int(100*self.df.loc[variate, col])
        sort_value = INDEX_MULT*sort_value + variate_index  \
          + INDEX_MULT
      sort_values[col] = sort_value
    #
    return pd.Series(sort_values)

  def plotMarginals(self, ser_sort_order=None,
      **plot_opts):
    """
    Does a heatmap of the marginals. X-axis is variates; y-axis are features.
        Values are probabilities.
    :param pd.Series ser_sort_order: Series with features as 
        index with floats defining order. 
    :param dict plot_opts:
    :return list-str: columns in sorted order
    """
    def setDefault(opts, key, value):
      if not key in opts.keys():
        opts[key] = value
    #
    if plot_opts is None:
      plot_opts = {}
    df = self.df.copy()
    pairs = zip(df.columns, self.calcSortIndex().tolist())
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    sorted_columns = [p[0] for p in sorted_pairs]
    df = df[sorted_columns]
    opts = dict(plot_opts)
    setDefault(opts, cn.PLT_XLABEL, "Variate")
    setDefault(opts, cn.PLT_YLABEL, "Feature")
    util_plots.plotCategoricalHeatmap(df.T, **opts)
    return sorted_columns

  def isSameColumns(self, other):
    """
    Checks that another MultipleDensity has the same columns.
    :param MultipleDensity other:
    """
    return set(other.df.columns) == set(self.df.columns)

  def isSameIndex(self, other):
    """
    Checks that another MultipleDensity has the same iindices.
    :param MultipleDensity other:
    """
    return set(other.df.index) == set(self.df.index)

  @staticmethod
  def _makeMarginalComparisonDF(ser_this, ser_other):
    """
    :param pd.Series this: probabilities indexed by gene
    :param pd.Series other: probabilities indexed by gene
    :return pd.DataFrame:
        columns: THIS, OTHER
        index: column
        values: float in [0, 1]
    """
    return pd.DataFrame({THIS: ser_this, OTHER: ser_other})

  def plotMarginalComparisons(self, other, ser_sort_order=None, 
      is_plot=True, **plot_opts):
    """
    Multiple line plots (one for each variate)
      x-axs probability of a in this distribution
      y-axis probability in other distribution
      point is order pairs of probabilities for the same feature and variate
    :param MultipleDensity other:
    :param list-object sort_order:
    """
    if not self.isSameColumns(other):
      raise ValueError("MultipleDensitys don't have the same columns")
    if not self.isSameIndex(other):
      raise ValueError("MultipleDensitys don't have the same index")
    variates = list(set(self.df.index))
    df_this = self.df.T
    df_other = other.df.T
    ax = None
    for idx, variate in enumerate(variates):
      df_plot = self.__class__._makeMarginalComparisonDF(
          df_this[variate], df_other[variate],)
      if ax is None:
        ax = df_plot.plot.scatter(THIS, OTHER, 
            color=COLORS[idx], **plot_opts)
      else:
        df_plot.plot.scatter(THIS, OTHER, ax=ax,
            color=COLORS[idx], **plot_opts)
    plt.legend(variates)
    ax.plot([0, 1.0], [0, 1.0], linestyle='dashed', color='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if is_plot:
      plt.show()
      
