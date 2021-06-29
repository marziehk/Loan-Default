import pandas as pd
import numpy as np
from Dim_Rd import *

df = pd.DataFrame({"feature1": [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                   "feature2": [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                   "feature3": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "feature4": [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                   "feature5": [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
                   "target": [1, 1, 1, 1, 1, 0, 0, 0, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3]})


# df = pd.DataFrame({"feature1": [1, None, 0, None, None, 1, None, 0, None, 1, None, 1, None, None, 0, None, None, 0, None, 1],
#                    "feature2": [None, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
#                    "feature3": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    "feature4": [None, 1, 0, None, 0, 0, 1, 1, None, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
#                    "feature5": [None, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
#                    "target": [None, 1, 1, 1, 1, 0, 0, None, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3]})
df_copy=df.copy()

df_reduce = Dim_Red(data=df, target="target")

print("Using DT:\n", df_reduce.DT(num_feature=2))
print("feature_scores:\n", df_reduce.feature_scores)

print("Using PCA:\n", df_reduce.PCA(n_components=2))
print("explained variance ratio:\n", df_reduce.explained_variance_ratio)
print("singular values:\n", df_reduce.singular_values)

print("Using LDA:\n", df_reduce.LDA(n_components=2))

##SVD
n_components = 2
df_svd = df_reduce.SvdTrunc (n_components)
print(f"Transformed Matrix after reducing to {n_components} features:\n", df_svd[0])
print("singular values:\n", df_reduce.singular_values)

#Deleting           For testing this part you need to comment the other parts and uncomment the second data frame
df_delete = df_reduce.delete(axis=0, percent=50)
print("Using deletion:\n", df_delete)

#Wrapper
df_reduce = Dim_Red(data=df_copy, target="target")
print("Using Wrapper:\n", df_reduce.wrapper(estimator=LogisticRegression(),forward=True,k_features=2,cv=0,scoring=None))

#Correlation
df_reduce = Dim_Red(data=df_copy, target="target")
print("Using Correlation:\n", df_reduce.corr(num_features=3))