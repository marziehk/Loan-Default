import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class Encoder():

    def __init__(self, *, data):
        self.data = data.copy()

    def GetDummies(self, categoryCol):

        for catCol in categoryCol:
            self.data = pd.concat([self.data, pd.get_dummies(self.data[catCol], prefix=catCol)], axis=1)
            self.data.drop([catCol], axis=1, inplace=True)
        return self.data

    def OneHotEncode(self, categoryCol):

        Enc_ohe, Enc_label = OneHotEncoder(), LabelEncoder()
        for catCol in categoryCol:
            encoder = Enc_label.fit_transform(self.data[catCol])
            dummiesCol = pd.DataFrame(Enc_ohe.fit_transform(encoder.squeeze().reshape(-1, 1)).todense(), columns=Enc_label.classes_)
            self.data.drop([catCol], axis=1, inplace=True)
            self.data = pd.concat([self.data, dummiesCol], axis=1)
        return self.data


    def LabelEncode(self, labelCol):

        Enc_label = LabelEncoder()
        self.data[labelCol] = Enc_label.fit_transform(np.array(self.data[labelCol]).reshape(-1))
        return self.data