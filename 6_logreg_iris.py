import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()

print(dir(iris))
# print(iris['data'][0])
# print(iris['feature_names'])
# print(iris['target'][0])
# print(iris['target_names'][0])

# =================================
# create dataframe

df = pd.DataFrame(
    iris['data'],
    columns = iris['feature_names']
)
df['target'] = iris['target']
df['jenis'] = df['target'].apply(
    lambda x: iris['target_names'][x]
)
print(df)

# ================================
# split train 90% & test 10%

from sklearn.model_selection import train_test_split
a, b, c, d = train_test_split(
    df[
        ['sepal length (cm)', 
        'sepal width (cm)', 
        'petal length (cm)', 
        'petal width (cm)']
    ],
    df['jenis'],
    test_size = .1
)

# ================================
# logistic regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', multi_class='auto')

# training model
model.fit(a, c)

# accuracy
print(model.score(a, c))

# prediction on b (x test)
# print(b.iloc[0:1])
# prediksi = model.predict(b.iloc[0:1])
# print(prediksi[0])
# print(d.iloc[0:1])

# prediction on my own data
prediksi = model.predict([[3,3,3,3]])
print(prediksi[0])