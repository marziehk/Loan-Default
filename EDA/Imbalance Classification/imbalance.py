import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import where
from matplotlib import pyplot
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

class imbalance():
    
    def __init__(self, data, target, features=None):
      
        self.target = target
        self.data = data.copy()
        self.features = list(set(self.data.columns) - set([self.target]))

class resampling(imbalance):

    def __init__(self, data, target, features=None):
        super().__init__(data, target, features)
        y = self.data[self.target]
        X = self.data[self.features]
        self.OverSampling = object_over(data, target)

class object_over(imbalance):

    def __init__(self, data, target, features=None):
      super().__init__(data, target, features)
      self.y = self.data[self.target].to_numpy()
      self.X = self.data[self.features].to_numpy()

    def Smote(self, sampling_strategy=None, k_neighbors=None):

      if not sampling_strategy: sampling_strategy = 'auto'
      if not k_neighbors: k_neighbors = 5
      oversample = SMOTE(sampling_strategy, k_neighbors)
      X_s, y_s = oversample.fit_resample(self.X, self.y)
      df_s1 = pd.DataFrame(X_s, columns = self.features)
      df_s2 = pd.DataFrame(y_s, columns = [self.target])
      self.data = pd.concat([df_s1, df_s2], axis=1)
      return self.data

    def CombSmoteUnder(self, sampling_strategy_Over=None, sampling_strategy_Under=None, k_neighbors=None):

      if not sampling_strategy_Over: sampling_strategy_Over = 'auto'
      if not k_neighbors: k_neighbors = 5
      if not sampling_strategy_Under: sampling_strategy_Under = 'auto'
      over = SMOTE(sampling_strategy_Over, k_neighbors)
      under = RandomUnderSampler(sampling_strategy_Under)
      steps = [('o', over), ('u', under)]
      pipeline = Pipeline(steps=steps)
      x_new, y_new = pipeline.fit_resample(self.X, self.y)
      df_c1 = pd.DataFrame(x_new, columns = self.features)
      df_c2 = pd.DataFrame(y_new, columns = [self.target])
      self.data = pd.concat([df_c1, df_c2], axis=1)
      return self.data

    def SmoteBord(self, sampling_strategy=None, k_neighbors=None):

      if not sampling_strategy: sampling_strategy = 'auto'
      if not k_neighbors: k_neighbors = 5
      Over_border = BorderlineSMOTE(sampling_strategy, k_neighbors)
      X_bord, y_bord = Over_border.fit_resample(self.X, self.y)
      df_b1 = pd.DataFrame(X_bord, columns = self.features)
      df_b2 = pd.DataFrame(y_bord, columns = [self.target])
      self.data = pd.concat([df_b1, df_b2], axis=1)
      return self.data

    def SmoteBordSVM(self, sampling_strategy=None, k_neighbors=None):

      if not sampling_strategy: sampling_strategy = 'auto'
      if not k_neighbors: k_neighbors = 5
      Over_border_svm = SVMSMOTE(sampling_strategy, k_neighbors)
      X_svm, y_svm = Over_border_svm.fit_resample(self.X, self.y)
      df_bs1 = pd.DataFrame(X_svm, columns = self.features)
      df_bs2 = pd.DataFrame(y_svm, columns = [self.target])
      self.data = pd.concat([df_bs1, df_bs2], axis=1)
      return self.data
    
    def SmoteAdap(self, ratio=None, n_neighbors=None):

      if not ratio: ratio = 'auto'
      if not n_neighbors: n_neighbors = 5
      oversample_ads = ADASYN(ratio, n_neighbors)
      X_adap, y_adap = oversample_ads.fit_resample(self.X, self.y)
      df_sa1 = pd.DataFrame(X_adap, columns = self.features)
      df_sa2 = pd.DataFrame(y_adap, columns = [self.target])
      self.data = pd.concat([df_sa1, df_sa2], axis=1)      
      return self.data

    def RandomSample(self, sampling_strategy=None):

      if not sampling_strategy: sampling_strategy = 'auto'
      Rand = RandomOverSampler(sampling_strategy)
      x_rand, y_rand = Rand.fit_resample(self.X, self.y)
      df_rand1 = pd.DataFrame(x_rand, columns = self.features)
      df_rand2 = pd.DataFrame(y_rand, columns = [self.target])
      self.data = pd.concat([df_rand1, df_rand2], axis=1) 
      return self.data

    def CounterTarget(self, data, target):

      if not target: target  = self.target
      y = data[target].to_list()
      counter = Counter(y)
      return counter