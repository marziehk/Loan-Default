# Loan-Default
This case study aims to identify patterns which indicate if a client has difficulty paying their installments which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc. This will ensure that the consumers capable of repaying the loan are not rejected.

# Missing Data Imputation using Regression

Complete Case Analysis, such as dropping missing values, should be rarely used since it might delete the other important data points containing the missing values which leads to smaller number of samples to train the learning model. In this case, the performace of the model would be degraded and accuracy would be a concern.
In addition, replacing the missing data with some common value such as mean and median of the dataset might under(over)estimate data. In other words, we add some bias to our estimation.

There are many missing data imputation methods to avoid these troublesome cases and Regression Imputation is one such method in which we estimate the missing values by Regression using other variables as the parameters.

