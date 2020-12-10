'''Constants used in Tellurium Models.'''

import os

# Paths
TELLURIUM_DIR = os.path.dirname(os.path.abspath(__file__))
MRNA_PATH = os.path.join(TELLURIUM_DIR, "wild.csv")


# DataFrame columns
MEAN = "mean"
STD = "std"
TIME = "time"

# Antimony file
HEAD = "head"
PROTEIN = "protein"
INITIALIZATIONS = "initializations"
CONSTANTS = "constants"
PARTS = [HEAD, PROTEIN, INITIALIZATIONS, CONSTANTS]
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DICT = {}
for part in PARTS:
  filename = "model_%s.txt" % part
  PATH_DICT[part] = os.path.join(CUR_DIR, filename)

# Column names
NAME = "name"
SORT = "sort"
VALUE = "value"

# Paths
