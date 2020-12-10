'''Utilities for Transforming DataFrames.'''

import pandas as pd
import numpy as np


def pruneSmallRows(df, min_abs):
  """
  Drops rows with an absolute value less than min_abs or are
  np.nan.
  :param pd.DataFrame df:
  :param float min_abs:
  :return pd.DataFrame:
  """
  indices = []
  for idx in df.index:
    if all([np.abs(v) < min_abs for v in df.loc[idx, :]
        if not np.isnan(v)]):
      indices.append(idx)
  return df.drop(indices, axis=0)
