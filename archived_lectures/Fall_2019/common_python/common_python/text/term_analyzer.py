'''Analyzes text terms in a DataFrame.'''

import common_python.constants as cn
from common_python.types.extended_list import ExtendedList

import pandas as pd
import numpy as np


NOISE_TERMS = ["the", "(", ")", "a", "and"]
SEPARATOR = " "


class TermAnalyzer(object):

  def __init__(self, noise_terms=NOISE_TERMS):
    if noise_terms is None:
      noise_terms = []
    self._noise_terms = noise_terms
    self.df_term = None

  @staticmethod
  def _removeAll(a_list, element):
    while a_list.count(element) > 0:
      a_list.remove(element)

  def makeSingleGroupDF(self, ser):
    """
    Constructs a dataframe of the terms found in a single group
    of genes.
    :param pd.Series ser:
      value: blank separated string of terms
    Updates self.df_term
      index: term
      COUNT - count of occurrences in a row
      FRAC - fraction of rows in which the term occurs
    """
    lines = []
    _ = [lines.append(s) for _, s in ser.items()]
    long_string = ' '.join(lines)
    all_terms = ExtendedList(long_string.split(SEPARATOR))
    for term in self._noise_terms:
      all_terms.removeAll(term)
    df = pd.DataFrame({
        cn.VALUE: all_terms,
        })
    dfg = df.groupby(cn.VALUE).size()
    self.df_term = pd.DataFrame(dfg)
    col = self.df_term.columns[0]
    self.df_term = self.df_term.rename(columns={col: cn.COUNT})
    self.df_term[cn.FRAC] = self.df_term[cn.COUNT]*1.0 / len(dfg)

  def makeMultipleGroupDF(self, df):
    """
    Constructs that a DF that summarizes results from multiple groups.
    :param pd.Series df:
      index: group identifier
      GENE_ID: gene
      TERMS: blank separated string of terms
    Updates self.df_term
      index: term
      COUNT - count of occurrences in a row
      FRAC - fraction of rows in which the term occurs
    """
    pass

  def plot(self, is_plot=True):
    """
    Constructs a bar plot of the terms present
    """
    pass
