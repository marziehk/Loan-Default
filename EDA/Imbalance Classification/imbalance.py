import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import RandomUnderSampler

class imbalance():

  def __init__(self, data, target, features=None, steps=[]):

    self.target = target
    self.data = data.copy()
    self.features = list(set(self.data.columns) - set([self.target]))
    self.target_numpy = self.data[self.target].to_numpy()
    self.features_numpy = self.data[self.features].to_numpy()
    self.steps = steps

  @staticmethod
  def resample_dataframe(features_resample, target_resample, features, target):

    df_features = pd.DataFrame(features_resample, columns=features)
    df_target = pd.DataFrame(target_resample, columns=[target])
    return pd.concat([df_features, df_target], axis=1)

class resampling(imbalance):

  def __init__(self, data, target, features=None, steps=[]):

    super().__init__(data, target, features, steps)
    self.OverSampling = object_over(self.data, self.target, steps=steps)
    self.UnderSampling = object_under(self.data, self.target, steps=steps)

    if not self.steps == []:
      pipeline = Pipeline(steps=self.steps)
      features_resample, target_resample = pipeline.fit_resample(self.features_numpy, self.target_numpy)
      self.data = self.resample_dataframe(features_resample=features_resample, target_resample=target_resample, features=self.features, target=self.target)
      self.resample = self.data

class object_under(imbalance):

  def __init__(self, data, target, features=None, steps=[]):

    super().__init__(data, target, features, steps)
    self.target_numpy = self.data[self.target].to_numpy()
    self.features_numpy = self.data[self.features].to_numpy()
    self.steps = steps

  def nearMiss(self, version=1, n_neighbors=3):

    under = NearMiss(version=version, n_neighbors=n_neighbors)
    self.steps.append(('u', under))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def condensedNearestNeighbour(self, n_neighbors=1):

    under = CondensedNearestNeighbour(n_neighbors=n_neighbors)
    self.steps.append(('u', under))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def tomekLinks(self):

    under = TomekLinks()
    self.steps.append(('u', under))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def editedNearestNeighbours(self, n_neighbors=3):

    under = EditedNearestNeighbours(n_neighbors=n_neighbors)
    self.steps.append(('u', under))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def oneSidedSelection(self, n_neighbors=1, n_seeds_S=200):

    under = OneSidedSelection(n_neighbors=n_neighbors, n_seeds_S=n_seeds_S)
    self.steps.append(('u', under))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def neighbourhoodCleaningRule(self, n_neighbors=3, threshold_cleaning=0.5):

    under = NeighbourhoodCleaningRule(n_neighbors=n_neighbors, threshold_cleaning=threshold_cleaning)
    self.steps.append(('u', under))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def underRandomSample(self, sampling_strategy='auto'):

    under = RandomUnderSampler(sampling_strategy=sampling_strategy)
    self.steps.append(('u', under))
    return resampling(data=self.data, target=self.target, steps=self.steps)

class object_over(imbalance):

  def __init__(self, data, target, features=None, steps=[]):

    super().__init__(data, target, features)
    self.target_numpy = self.data[self.target].to_numpy()
    self.features_numpy = self.data[self.features].to_numpy()
    self.steps = steps

  def Smote(self, sampling_strategy='auto', k_neighbors=5):

    over = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    self.steps.append(('o', over))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def SmoteBord(self, sampling_strategy='auto', k_neighbors=5):

    over = BorderlineSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    self.steps.append(('o', over))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def SmoteBordSVM(self, sampling_strategy='auto', k_neighbors=5):

    over = SVMSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    self.steps.append(('o', over))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def SmoteAdap(self, sampling_strategy='auto', n_neighbors=5):

    over = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors)
    self.steps.append(('o', over))
    return resampling(data=self.data, target=self.target, steps=self.steps)

  def overRandomSample(self, sampling_strategy='auto'):

    over = RandomOverSampler(sampling_strategy=sampling_strategy)
    self.steps.append(('o', over))
    return resampling(data=self.data, target=self.target, steps=self.steps)

