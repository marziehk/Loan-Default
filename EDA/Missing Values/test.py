import pandas as pd
import missing

df = pd.DataFrame({"feature1":[1, 57, 23, 7, 8, 0, None, 54, None, None],
                   "feature2": [1, 2, 3, 4, 5, 8, 7, 8, 9, 10],
                   "feature3": [3, 67, 89, 32, 6, 78, 33, 6, 1, None],
                   "label": [4, 3, 76, 54, 26, 46, 3, 5, 6, 9]})

Missing_Features = ["feature1","feature3"]

df_missing = missing.missingvalue(data=df, missCol=Missing_Features)
# df_filled = df_missing.regression_random()
# df_filled = df_missing.regression_random(parameters=["feature2"])

print("Orignial Dataset: \n", df)
print("Cleaned Dataset: \n", df_filled)


