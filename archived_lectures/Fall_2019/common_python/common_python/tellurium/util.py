'''Utilities used in Tellurium.'''

import pandas as pd
import numpy as np

def dfToSer(df):
  """
  Converts a dataframe to a series.
  :param pd.DataFrame df:
  :return pd.Series:
  """
  return pd.concat([df[c] for c in df.columns])

def isEqualList(list1, list2):
  return len(set(list1).symmetric_difference(list2)) == 0

def readFile(path):
  with open(path, "r") as fd:
    result = fd.readlines()
  return "\n".join(result)

def calcRsq(ser_obs, ser_est):
  ser_res = ser_obs - ser_est
  ser_res = ser_res.dropna()
  return 1 - ser_res.var() / ser_obs[ser_res.index].var()

def interpolateTime(ser, time):
  """
  Interpolates a values between two times.
  :param pd.Series ser: index is time
  :param float time:
  :return float:
  """
  def findTime(a_list, func):
    if len(a_list) == 0:
      return np.nan
    else:
      return func(a_list)
  def findValue(time):
    if np.isnan(time):
      return np.nan
    else:
      return ser[time]
  #
  time_lb = findTime([t for t in ser.index if t <= time], max)
  time_ub = findTime([t for t in ser.index if t >= time], min)
  value_lb = findValue(time_lb)
  value_ub = findValue(time_ub)
  if np.isnan(value_lb):
    return value_ub
  if np.isnan(value_ub):
    return value_lb
  if time_ub == time_lb:
    return value_ub
  frac = (time - time_lb)/(time_ub - time_lb)
  return (1 - frac)*value_lb + frac*value_ub

def makeTimeInterpolatedMatrix(df, num_interpolation=10):
  """
  Does linear interpolations of values based on time.
  :param pd.DataFrame df: index is time
  :param int num_interpolation: number of interpolations between time
  :return np.array: first column is time
  Assumes that index is sorted ascending
  """
  times = df.index.tolist()
  time_last = times[0]
  matrix = []
  # For each pair of times
  for time in times[1:]:
    time_incr = (time - time_last)/num_interpolation
    arr_last = np.array(df.loc[time_last, :])
    arr_cur = np.array(df.loc[time, :])
    arr_incr = (arr_cur - arr_last)/num_interpolation
    # For each interpolation
    for idx in range(num_interpolation):
      arr = arr_last + idx*arr_incr
      arr = np.insert(arr, 0, time_last + idx*time_incr)
      matrix.append(arr)
    time_last = time
  return np.array(matrix)
