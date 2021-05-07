import pandas as pd
from sklearn.datasets import make_classification
from collections import Counter
from imbalance import *


X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

df = pd.DataFrame({'feature1': X[:,0], 'feature2': X[:,1], 'target': y})
target = 'target'

df_imbalance = resampling(data=df, target=target)

# df_resample = df_imbalance.OverSampling.Smote().resample
# df_resample = df_imbalance.OverSampling.SmoteBord().resample
# df_resample = df_imbalance.OverSampling.SmoteBordSVM().resample
# df_resample = df_imbalance.OverSampling.SmoteAdap().resample
# df_resample = df_imbalance.OverSampling.overRandomSample().resample
# df_resample = df_imbalance.UnderSampling.nearMiss(version=1, n_neighbors=3).resample
# df_resample = df_imbalance.UnderSampling.condensedNearestNeighbour(n_neighbors=1).resample
# df_resample = df_imbalance.UnderSampling.tomekLinks().resample
# df_resample = df_imbalance.UnderSampling.editedNearestNeighbours().resample
# df_resample = df_imbalance.UnderSampling.oneSidedSelection().resample
# df_resample = df_imbalance.UnderSampling.neighbourhoodCleaningRule().resample
# df_resample = df_imbalance.UnderSampling.underRandomSample().resample
# df_resample = df_imbalance.UnderOver.CombSmoteUnder().resample
df_resample = df_imbalance.OverSampling.overRandomSample(sampling_strategy=0.1).UnderSampling.underRandomSample(sampling_strategy=0.5).resample


count_before = Counter(df[target].to_list())
count_after = Counter(df_resample[target].to_list())

print(count_before)
print(count_after)
