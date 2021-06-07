import pandas as pd
import numpy as np
from encoding import Encoder

df = pd.DataFrame({"feature1":[1, 57, 23, 7, 8, 0, 11, 54, 0, 1],
                   "feature2": [1, 2, 3, 4, 5, 8, 7, 8, 9, 10],
                   "feature3": ['Red', 'Blue', 'Red', 'Yellow', 'Blue', 'Blue', 'Yellow', 'Red', 'Yellow', 'Red'],
                   "feature4": ['France', 'USA', 'USA', 'Canada', 'USA', 'Canada', 'France', 'Canada', 'USA', 'France'],
                   "label":['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']})

#categoryCol = ["feature3", "feature4"]

labelCol = ['label']
categoryCol= [x for x in df.columns.drop(labelCol) if df[x].dtype == 'object']




df_category = Encoder(data=df)
# df_encoded = df_category.GetDummies(categoryCol=categoryCol)
df_encoded = df_category.OneHotEncode(categoryCol=categoryCol)
df_encoded = df_category.LabelEncode(labelCol=labelCol)

print("Orignial Dataset: \n", df)
print("Encoded Dataset: \n", df_encoded)

