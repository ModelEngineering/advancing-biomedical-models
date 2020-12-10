"""
Encapsulate Random forest as a ClassifierEnsemble.
This provides uniform handling of classifiers. Also, provides:
1. predictions are probabilistic
2. access to plotting
"""

import common_python.constants as cn
from common_python.util import util
from common_python.classifier.classifier_collection  \
    import ClassifierCollection
from common_python.classifier.classifier_ensemble  \
    import ClassifierEnsemble, ClassifierDescriptor

import collections
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RF_ESTIMATORS = "n_estimators"
RF_MAX_FEATURES = "max_features"
RF_BOOTSTRAP = "bootstrap"
RF_DEFAULTS = {
    RF_ESTIMATORS: 500,
    RF_BOOTSTRAP: True,
    }

  
##########################################################
# TODO: classmethod that makes RF with defaults
class ClassifierDescriptorRandomForest(ClassifierDescriptor):
  # Descriptor information needed for SVM classifiers
  # Descriptor is for one-vs-rest. So, there is a separate
  # classifier for each class.
  
  def __init__(self, clf=RandomForestClassifier()):
    self.clf = clf

  def getImportance(self, _, **__):
    """
    Calculates the importances of features.
    :return list-float:
    """
    return self.clf.feature_importances_

  
##########################################################
class ClassifierEnsembleRandomForest(ClassifierEnsemble):

  def __init__(self, 
      clf_desc=ClassifierDescriptorRandomForest(),
      size=20,
      **kwargs):
    """
    :param ClassifierDescriptor clf_desc:
    :param dict kwargs: arguments passed to classifier
    """
    super().__init__(clf_desc=clf_desc, size=size, **kwargs)

  def fit(self, df_X, ser_y):
    def collectionMaker(df_X, ser_y):
      self.clf_desc.clf.fit(df_X, ser_y)
      classes = ser_y.unique().tolist()
      classes.sort()
      return ClassifierCollection(
          clfs=[self.clf_desc.clf],
          features=df_X.columns.tolist(),
          classes=classes)
    super().fit(df_X, ser_y, collectionMaker=collectionMaker)
        
    self.classes.sort()

  # TODO: Create true probabilistic prediction by running
  #       individual trees
  def predict(self, df_X):
    """
    Constructs the prediction dataframe for RF
    :param pd.DataFrame df_X: feature fectors (rows)
    """
    width = len(self.classes)
    length = len(df_X)
    row = np.repeat(0, width)
    data = np.repeat(row, length)
    data.resize(length, width)
    df = pd.DataFrame(data)
    df.columns = self.classes
    df.index = df_X.index
    #
    predictions = self.clf_desc.clf.predict(df_X)
    for idx, prediction in zip(df.index, predictions):
      for column in df.columns:
        if column == prediction:
          df.loc[idx, column] = 1.0
    return df
