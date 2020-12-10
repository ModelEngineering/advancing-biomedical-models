"""Generates Data from an Empirical Distribution."""


import pandas as pd
import numpy as np
import scipy.stats as stats

import common_python.constants as cn
from common_python.statistics import util_statistics
from common_python.plots import util_plots


class EmpiricalDistributionGenerator(object):
  """
  DataFrames are structured so that columns are features and
  rows are instances (observations).
  Useage:
    1. Simple data.
       generator = EmpiricalDistributionGenerator(df)
       df_data = generator.sample(nobs)
    2. Data with replacing values.
       generator = EmpiricalDistributionGenerator(df)
       df_data = generator.synthesize(nobs, frac)
  """

  def __init__(self, df):
    """
    :param pd.DataFrame: empirical distribution
    """
    self._df = df

  def sample(self, nobs, is_decorrelate=True):
    """
    Samples with replacement.
    :param int nobs:
    """
    df_sample = self._df.sample(nobs, replace=True)
    if is_decorrelate:
      df_sample = util_statistics.decorrelate(df_sample)
    df_sample.index = range(len(df_sample))
    return df_sample

  def synthesize(self, nobs, frac):
    """
    Returns a random sample of rows with frac values replaced
    by values from the empirical CDF.
    :param int nobs: number of observations in the sample
        if nobs < 0, use self._df without sampling
    :param float frac: fraction of values replaced
    :return pd.DataFrame:
    """
    if nobs < 0:
      is_sample = False
    else:
      is_sample = True
    # Generate base data
    ncols = len(self._df.columns)
    # Get a sample of rows, preserving the covariance structure
    if is_sample:
      df_correlated = self.sample(nobs, is_decorrelate=False)
    else:
      df_correlated = self._df.copy()
      df_correlated.index = range(len(df_correlated))
      nobs = len(self._df)
    # Draw from the marginals without the covariance structure
    df_uncorrelated = self.sample(nobs, is_decorrelate=True)
    # Determine values to replace
    df_replace = pd.DataFrame(stats.bernoulli.rvs(frac,
         size=(nobs, ncols)))
    df_replace.columns = self._df.columns
    df_keep = 1 - df_replace
    # Create the replacement data
    df_new = df_correlated*df_keep + df_uncorrelated*df_replace
    return df_new
