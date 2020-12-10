'''Utility routines.'''

import os
import random
import string
import sys
import pandas as pd
import numpy as np

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def ConvertType(v):
  # Converts to int, float, str as required
  # Input: v - string representation
  # Output: r - new representation
  try:
    r = int(v)
  except:
    try:
      r = float(v)
    except:
      r = v  # Leave as string
  return r

def ConvertTypes(values):
  # Converts a list strings to a list of their types
  # Input: values - list
  # Output: results
  results = []
  for v in values:
    results.append(ConvertType(v))
  return results

def randomWords(count, size=5):
  # Generates a sequence of random words of the same size
  # Input: count - number of random words generated
  #        size - size of each word
  # Output: result - list of random words
  return [randomWord(size=size) for n in range(count)]

def randomWord(size=5):
  # Generates a random word
  # Input: size - size of each word
  # Output: word
  word = ''
  for n in range(size):
    word += random.choice(LETTERS)
  return word

# TODO: Add tests
def stringToClass(cls_str):
  """
  Converts the string representation of a class to a class object.
  :param str cls_str: string representation of a class
  :return type class:
  """
  import_stg1 = cls_str.split(" ")[1]
  import_stg2 = import_stg1.replace("'", "")
  import_stg3 = import_stg2.replace(">", "")
  import_parse = import_stg3.split(".")
  cls = import_parse[-1]
  import_path = '.'.join(import_parse[:-1])
  import_statement = "from %s import %s" % (import_path, cls)
  exec(import_statement)
  assign_statement = "this_class = %s" % cls
  exec(assign_statement)
  return this_class

def getFileExtension(filepath):
  """
  :param str filepath:
  :return str: extension excluding the "."
  """
  extension = os.path.split(filepath)[-1]
  split_filename = extension.split('.')
  if len(split_filename) == 1:
    ext = None
  else:
    ext = split_filename[-1]
  return ext

def stripFileExtension(filepath):
  """
  :param str filepath:
  :return str:
 filepath without the extension
  """
  split_filepath = list(os.path.split(filepath))
  filename = split_filepath[-1]
  split_filename = filename.split('.')
  stripped_filename = split_filename[0]
  split_filepath[-1] = stripped_filename
  fullpath = ""
  for ele in split_filepath:
    fullpath = os.path.join(fullpath, ele)
  return fullpath

def changeFileExtension(filepath, extension):
  """
  :param str filepath:
  :param str extension: without "."
  :return str: filepath without the extension
  """
  stripped_filepath = stripFileExtension(filepath)
  if extension is None:
    return "%s" % stripped_filepath
  else:
    return "%s.%s" % (stripped_filepath, extension)

def getValue(dictionary, key, value):
  """
  Returns the value for the key in the dictionary or the default.
  :param dict dictionary:
  :param object key:
  :param object value:
  :return object:
  """
  if not key in dictionary.keys():
    return value
  else:
    return dictionary[key]

def setValue(dictionary, key, default_value):
  """
  Returns an updated dictionary set to the default value if
  none is present
  :param dict dictionary:
  :param object key:
  :param object default_value:
  :return dictionary:
  """
  value = getValue(dictionary, key, default_value)
  new_dict = dict(dictionary)
  new_dict[key] = value
  return new_dict

def setList(value):
  """
  Sets a list to empty if None.
  """
  if value is None:
    return []
  else:
    return value

def addPath(repo_name, sub_dirs=None):
  """
  Adds a path relative to the repository root.
  :param str repo_name:
  :param list-str sub_dirs:
  """
  if sub_dirs is None:
    sub_dirs = []
  path = os.path.dirname(os.path.abspath(__file__))
  done = False
  found_first_folder = False
  while not done:
    new_path, cur_folder  = os.path.split(path)
    if len(path) == 0:
      raise ValueError("Repo %s not found." % repo_name)
    if cur_folder == repo_name:
      if found_first_folder:
        root_folder = path
        done = True
        break
      else:
        found_first_folder = True
    path = new_path
  if not done:
    raise ValueError("Repository root of %s not found" % repo_name)
  for folder in sub_dirs:
    path = os.path.join(path, folder)
  sys.path.insert(0, path)

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
