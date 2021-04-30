import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class missingvalue():

    def __init__(self, *, data, missCol):
        self.missCol = missCol
        self.data = data.copy()

    @staticmethod
    def random_imputation(data, features):
        number_missing = data[features].isnull().sum()
        observed_values = data.loc[data[features].notnull(), features]
        data.loc[data[features].isnull(), features + '_imp'] = np.random.choice(observed_values, size=number_missing, replace = True)
        return data

    def random_imp(self):
        for feature in self.missCol:
            self.data[feature + '_imp'] = self.data[feature]
            self.data = self.random_imputation(self.data, feature)
        return self.data

    def regression_random(self, parameters=None):
        deter_data = pd.DataFrame(columns = ["Det_" + name for name in self.missCol])
        self.data = self.random_imp()
        for feature in self.missCol:
            if not parameters:
                parameters = list(set(self.data.columns) - set(self.missCol) - {feature + '_imp'})
            deter_data["Det_" + feature] = self.data[feature + "_imp"]
            #Create a Linear Regression model to estimate the missing data
            model = linear_model.LinearRegression()
            model.fit(X=self.data[parameters], y=self.data[feature + '_imp'])
            # Observe that I preserve the index of the missing data from the original dataframe
            deter_data.loc[self.data[feature].isnull(), "Det_" + feature] = model.predict(self.data[parameters])[self.data[feature].isnull()]
        return pd.concat([self.data, deter_data], axis=1)

        #return deter_data

    def iterative(self, parameters=None):
        imp_mean = IterativeImputer(random_state=0)
        for feature in self.missCol:
            if not parameters:
                parameters = list(set(self.data.columns) - set(feature))
            imp_mean.fit(np.array(self.data[parameters]).reshape(-1,1))
            self.data[feature] = imp_mean.transform(np.array(self.data[feature]).reshape(-1,1))
        return self.data

    def backward(self):
        for feature in self.missCol:
            self.data[feature] = self.data[feature].fillna(method='bfill').fillna(method='ffill')
        return self.data

    def forward(self):
        for feature in self.missCol:
            self.data[feature] = self.data[feature].fillna(method='ffill').fillna(method='bfill')
        return self.data

    def interpolate(self):
        for feature in self.missCol:
            self.data[feature].interpolate(method='values', inplace = True, limit_direction ="both")
        return self.data

    def replace(self, kind):
        if kind == 'median':
            median = self.data[self.missCol].median()
            self.data.fillna(value = median, inplace= True)

        if kind == 'mode':
            mode = self.data[self.missCol].mode()
            self.data.fillna(value = mode.iloc[0], inplace= True)  # mode can be multiple numbers, so we choose the first one

        if kind == 'mean':
            mean = self.data[self.missCol].mean()
            self.data.fillna(value = mean, inplace= True)

        return self.data

    def deletion(self, *, axis=None):
        if not axis or axis == 'rows' or axis == 0:
            self.data.dropna(subset = self.missCol, inplace=True)
        if axis == 'columns' or axis == 1:
            self.data.drop(self.missCol, inplace=True, axis='columns')
        return self.data