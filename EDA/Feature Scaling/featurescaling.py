import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
class featurescaling():

    def __init__(self,*,data,kind):
        self.data=data.copy()
        self.kind=kind
    def scaling(self):
        if self.kind=="MinMaxScaler":
            # define min max scaler
            scaler=MinMaxScaler()
            # transform data
            self.data=scaler.fit_transform(self.data)
        if self.kind=="StandardScaler":
            # define standars scaler
            scaler=StandardScaler()
            # transform data
            self.data=scaler.fit_transform(self.data)
        return self.data