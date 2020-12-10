'''Analyzes collections of ordinal values.'''
"""
An ordinal value (ovalue) is an object with a specified order
relative to other objects in its collection. The default
order is based on position in the collection.

Ordinal collections can be compared in the following ways:
1. overlap. The fraction of the values of both that are shared.
            |A interesection B| / |A union B|
2. ordering. The fraction of pairwise comparisons of weights
             of categorical values that are the same in both
             categorical collections.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OrdinalCollection(object):

  def __init__(self, ordinals):
    """
    :param list-object ordinal_values: ordinals
        values are ordered from smallest to largest
        in ordinal value
    """
    self.ordinals = list(ordinals)

  @classmethod
  def makeWithOrderings(cls, ordinals, orderings, is_abs=True):
    """
    Creates an OrdinalCollection using the list of orderings.
    :param list-list-float orderings: the inner list has
        one value for each ordinal
    :param bool is_abs: use absolute value
    :return OrdinalCollection:
    """
    if is_abs:
      func = np.abs
    else:
      func = lambda v: v  # identity function
    # Terms in order sorted by weight
    adjusted_orderings = [[func(x) for x in xv] for xv in orderings]
    keys = [max(xv) for xv in zip(*adjusted_orderings)]
    sorted_ordinals = [o for _, o in sorted(zip(keys, ordinals))]
    return cls(sorted_ordinals)

  @staticmethod
  def _calcTopN(ordinals, top):
    if top is None:
      top = len(ordinals)
    return ordinals[-top:]

  def compareOverlap(self, others, topN=None):
    """
    Calculates the overlap between two OrdinalCollections.
    Computes the ratio of the size of intersection to the size
    of the union.
    :param list-OrdinalCollection other:
    :return float: fraction of overlap
    """
    cls = self.__class__
    #
    ordinal_sets = [set(cls._calcTopN(self.ordinals, topN))]
    for other in others:
      ordinal_sets.append(set(cls._calcTopN(other.ordinals, topN)))
    ordinal_union = set([])
    for other in ordinal_sets:
      ordinal_union = ordinal_union.union(other)
    ordinal_intersection = ordinal_union
    for other in ordinal_sets:
      ordinal_intersection = ordinal_intersection.intersection(other)
    num_intersection = len(ordinal_intersection)
    num_union = len(ordinal_union)
    return (1.0*num_intersection)/num_union

  def makeOrderMatrix(self, top=None):
    """
    Create a matrix where a 1 in cell ij means that ordinal
    i is less than ordinal j.
    :return pd.DataFrame:
    """
    ordinals = self.__class__._calcTopN(self.ordinals, top)
    length = len(ordinals)
    dfs = [pd.Series(np.repeat(0, length)) for _ in range(length)]
    df = pd.concat(dfs, axis=1)
    df.columns = ordinals
    df.index = ordinals
    for idx, ordinal in enumerate(ordinals):
      for pos in range(idx+1, length):
        df.loc[ordinal, df.columns[pos]] = 1
    return df
    
  def compareOrder(self, others, top=None):
    """
    Calculates the similarities in ordering of two
    OrdinalCollection.
    :param list-OrdinalCollection others:
    :return float: fraction of order preserved
    """
    df_matrix = self.makeOrderMatrix(top)
    all_ordinals = set(df_matrix.columns)
    for other in others:
      df = other.makeOrderMatrix(top)
      all_ordinals = all_ordinals.union(df.columns)
      df_matrix = df_matrix * other.makeOrderMatrix(top)
    #
    max_satisfied_inequalities =  \
        (len(all_ordinals)-1)*(len(all_ordinals))/2.0
    num_satisfied = df_matrix.sum().sum()
    return (1.0*num_satisfied)/max_satisfied_inequalities
