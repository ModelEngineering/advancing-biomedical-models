"""Helpers for classifier testing."""

from common_python.util.persister import Persister
#from common.trinary_data import TrinaryData

import os
import pandas as pd
import numpy as np


DIR_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_PATH = os.path.join(DIR_PATH,
    "test_classifier_data.pcl")
PERSISTER = Persister(TEST_DATA_PATH)


if not PERSISTER.isExist():
  DATA = TrinaryData()  # Will get an error if pcl not present
  PERSISTER.set(DATA)
else:
  DATA = PERSISTER.get()
  
def getData():
  """
  Provides classification data
  """
  df_X = DATA.df_X
  df_X.columns = DATA.features
  ser_y = DATA.ser_y
  return df_X, ser_y
