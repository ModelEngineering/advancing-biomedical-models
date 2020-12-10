"""Creation, Analysis, and Manipulation of Discrete Distributions."""

"""
TODO
1. Bar plot of density
"""


import common_python.constants as cn
from common_python.plots import util_plots

import collections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INDEX_MULT = 100


class Density(object):
  """
  self.ser_density is a Series whose index is the variate
  and the value is its density.
  """

  def __init__(self, ser, variates=None):
    """
    :param pd.Series: variate values for which a density is created
    :param list-object: expected values in the density
    """
    if variates is None:
      variates = ser.unique().tolist()
    self.variates = variates
    self.ser_density = self.__class__._makeDensity(ser, self.variates)

  @staticmethod
  def _makeDensity(ser, variates):
    """
    :param pd.Series ser:
    :param list-object variates: required to be present in result
    :return pd.Series:
    """
    counter = dict(collections.Counter(ser))
    length = len(ser)
    counter = {k: v/length for k, v in counter.items()}
    for variate in variates:
      if not variate in counter.keys():
        counter[variate] = 0
    values = list(counter.values())
    keys = list(counter.keys())
    return pd.Series(values, index=keys)

  def get(self):
    return self.ser_density

  # TODO: Write tests
  def isLessEqual(self, other):
    """
    Determines if lower values have higher probabilities.
    :param Density other:
    :return bool:
    """
    is_less = True
    for key in self.ser_density.keys():
      if is_less:
        if self.ser_density.loc[key][0] >  \
            other.ser_density.loc[key][0]:
          is_less = False
      else:
        if self.ser_density.loc[key][0] <  \
            other.ser_density.loc[key][0]:
          return False
    return True

  def plot(self, is_plot=True, **kwds):
    """
    Creates a bar plot of the density.
    """
    self.ser_density.plot.bar(**kwds)
    if is_plot:
      plt.show()
