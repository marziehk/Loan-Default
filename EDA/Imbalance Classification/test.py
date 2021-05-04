from sklearn.datasets import make_classification
import pandas as pd
from imbalance import *

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

df = pd.DataFrame({'feature1': X[:,0], 'feature2': X[:,1], 'target': y})
df_imbalance = resampling(data = df, target='target')

df_resample = df_imbalance.OverSampling.Smote()
# df_resample = df_imbalance.OverSampling.CombSmoteUnder()
# df_resample = df_imbalance.OverSampling.SmoteBord()
# df_resample = df_imbalance.OverSampling.SmoteBordSVM()
# df_resample = df_imbalance.OverSampling.SmoteAdap()
# df_resample = df_imbalance.OverSampling.RandomSample()

count =  df_imbalance.OverSampling.CounterTarget(df_resample, target = 'target')

print(count)