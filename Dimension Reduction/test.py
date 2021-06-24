import pandas as pd
import numpy as np
from Dim_Rd import *

df = pd.DataFrame({"feature1": [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                   "feature2": [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                   "feature3": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "feature4": [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                   "feature5": [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
                   "target": [1, 1, 1, 1, 1, 0, 0, 0, 4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3]})

df_reduce = Dim_Red(data=df, target="target")

print("Using DT:\n", df_reduce.DT(num_feature=2))
print("feature_scores:\n", df_reduce.feature_scores)

print("Using PCA:\n", df_reduce.PCA(n_components=2))
print("explained variance ratio:\n", df_reduce.explained_variance_ratio)
print("singular values:\n", df_reduce.singular_values)

print("Using LDA:\n", df_reduce.LDA(n_components=2))

