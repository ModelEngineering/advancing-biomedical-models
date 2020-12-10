"""Calculates statistics."""

import numpy as np
import pandas as pd
import scipy.stats as stats


def filterZeroVarianceRows(df):
  """
  Removes rows that have zero variance.
  :param pd.DataFrame df: numeric rows
  :return pd.DataFrame: same columns as df
  """
  ser_std = df.std(axis=1)
  indices = [i for i in ser_std.index 
      if np.isclose(ser_std[i], 0.0)]
  return df.drop(indices)

def calcLogSL(df, round_decimal=4, is_nan=True):
  """
  Calculates the minus log10 of significance levels
  for large values in rows.
  :param pd.DataFrame df: Rows have same units
  :param int round_decimal: How many decimal points to round
  :param bool is_nan: force std = 0 to be np.nan
  :return pd.DataFrame: transformed
  """
  ser_std = df.std(axis=1)
  ser_mean = df.mean(axis=1)
  df1 = df.subtract(ser_mean, axis=0)
  df2 = df1.divide(ser_std, axis=0)
  # Calculate minus log10 of significance levels
  # For efficiency reasons, this is done by dictionary lookup
  df3 = df2.fillna(-4.0)
  df3 = df3.applymap(lambda v: np.round(v, 4))
  values = []
  [ [values.append(v) for v in df3[c]] for c in df3.columns]
  vals_lookup = {v: -np.log10(1 - stats.norm(0, 1).cdf(v))
      for v in pd.Series(values).unique()}
  df4 = df3.applymap(lambda v: vals_lookup[v])
  if is_nan:
    df_log = df4.applymap(lambda v: v if v > 0.01 else np.nan)
  else:
    df_log = df4
  return df_log

def decorrelate(df):
  """
  Permutes rows within columns to remove 
      correlations between features.
  :return pd.DataFrame:
  """
  length = len(df)
  df_result = df.copy()
  for col in df_result.columns:
    values = df_result[col].tolist()
    df_result[col] = np.random.permutation(values)
  return df_result
