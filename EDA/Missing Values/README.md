# Missing Data Imputation using Regression:

Complete Case Analysis, such as dropping missing values, should be rarely used since it might delete the other important data points containing the missing values which leads to smaller number of samples to train the learning model. In this case, the performace of the model would be degraded and accuracy would be a concern. In addition, replacing the missing data with some common value such as mean and median of the dataset might under(over)estimate data. In other words, we add some bias to our estimation.

There are many missing data imputation methods to avoid these troublesome cases and Regression Imputation is one such method in which we estimate the missing values by Regression using other variables as the parameters.

First we initially impute all the variables with missing values using some trivial methods like # Simple Random Imputation (we impute the missing data with random observed values of the variable) which is later followed by Regression Imputation of each of the variables iteratively.

Then a Deterministic Regression Imputation is used to replace the missing data with the values predicted in our regression model and repeat this process for each variable. We believe this method has some advantages as it randomly choose some data to fill the missing value and it use regressing using those data to predict the missing values so it will add bias/noise to our model.

# Missing Data Imputation using IterativeImputer:

A more sophisticated approach is to use the IterativeImputer class, which models each feature with missing values as a function of other features, and uses that estimate for imputation. It does so in an iterated round-robin fashion: at each step, a feature column is designated as output y and the other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y. Then, the regressor is used to predict the missing values of y. This is done for each feature in an iterative fashion, and then is repeated for max_iter imputation rounds. The results of the final imputation round are returned. More information can be found in the link below: [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)