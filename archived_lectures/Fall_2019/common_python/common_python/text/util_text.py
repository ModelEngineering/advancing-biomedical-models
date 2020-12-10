'''Utilities for Text Analysis'''

import pandas as pd
import numpy as np


GROUP = "group"
TERM = "term"
COUNT = "count"


def makeTermMatrix(ser):
  """
  Creates a matrix terms counts. Row index is group; columns are terms.
  :param pd.Series ser:
    index: element identifier
    value: terms (treated as a string)
  """
  df = pd.DataFrame({
      GROUP: ser.index,
      TERM: ser.tolist(),
      COUNT: np.repeat(1, len(ser)),
      })
  dfg = df.groupby([GROUP, TERM]).sum()
  df1 = dfg.reset_index()
  df_result = df1.pivot(index=GROUP, columns=TERM, values=COUNT)
  df_result = df_result.applymap(lambda x: 0 if np.isnan(x) else x)
  return df_result
