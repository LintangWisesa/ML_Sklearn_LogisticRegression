import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()

print(dir(digits))
# print(digits['data'][0])
# print(digits['images'][0])
# print(digits['target'][0])
# print(digits['target_names'][0])

# ================================
# plot data

# fig = plt.figure('Data Digit', figsize=(6,6))
# plt.imshow(digits['images'][1200], cmap='gray')
# plt.title('Angka = {}'.format(digits['target'][1200]))
# plt.show()

# ================================
# split train 90% & test 10%

from sklearn.model_selection import train_test_split
a, b, c, d = train_test_split(
    digits['data'], 
    digits['target'],
    test_size = .1
)

# print(len(a))
# print(len(b))
# print(len(c))
# print(len(d))

# ================================
# logistic regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', multi_class='auto')

# training
model.fit(a, c)

# prediksi
print(b[0])
# print(b[0].reshape(1, -1))
# print(b[0].reshape(8, 8))
prediksi = model.predict(b[0].reshape(1, -1))
print(prediksi[0])
print(d[0])

# accuracy
accuracy = model.score(a, c)
print(accuracy)

# ================================
# plot data test & prediction

fig = plt.figure('Data Digit', figsize=(6,6))
plt.imshow(b[0].reshape(8,8), cmap='gray')
plt.title(
    'Prediction = {} | Actual = {} | Accuracy = {}%'
    .format(prediksi[0], d[0], accuracy*100)
)
plt.show()