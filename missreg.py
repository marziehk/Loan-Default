class missingvaluereg():
    def __init__(self,missing_columns,data):
        self.missing_columns=missing_columns
        self.data=data

    @staticmethod
    def random_imputation(data, features):

        number_missing = data[features].isnull().sum()
        observed_values = data.loc[data[features].notnull(), features]
        data.loc[data[features].isnull(), features + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
        return data

    def random_imp(self):
        for feature in self.missing_columns:
            self.data[feature + '_imp'] = self.data[feature]
            self.data = self.random_imputation(self.data, feature)
        return self.data
    def regressionpart(self):
        deter_data = pd.DataFrame(columns = ["Det" + name for name in self.missing_columns])
        for feature in self.missing_columns:

            deter_data["Det" + feature] = self.data[feature + "_imp"]
            parameters = list(set(self.data.columns) - set(self.missing_columns) - {feature + '_imp'})

            #Create a Linear Regression model to estimate the missing data
            model = linear_model.LinearRegression()
            model.fit(X = self.data[parameters], y = self.data[feature + '_imp'])

            #observe that I preserve the index of the missing data from the original dataframe
            deter_data.loc[self.data[feature].isnull(), "Det" + feature] = model.predict(self.data[parameters])[self.data[feature].isnull()]
        return pd.concat([self.data,deter_data],axis=1)
        #return deter_data