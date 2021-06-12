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

print(df_reduce.DT(num_feature = 2))

print(df_reduce.feature_scores)
