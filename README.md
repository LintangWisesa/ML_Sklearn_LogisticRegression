![simplinnovation](https://4.bp.blogspot.com/-f7YxPyqHAzY/WJ6VnkvE0SI/AAAAAAAADTQ/0tDQPTrVrtMAFT-q-1-3ktUQT5Il9FGdQCLcB/s350/simpLINnovation1a.png)

# Sklearn: Logistic Regression

### **Basic Formula**

In statistics, __logistic regression__ is a predictive analysis that used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. Here is the basic formula of logistic regression:

![Rumus Logistic Regression](http://faculty.cas.usf.edu/mbrannick/regression/gifs/lo8.gif)

#

### **Using Microsoft Excel**

Here is the example of linear regression using Microsoft Excel. Clone/download this repo & open file: __*0_logisticRegression.xlsx*__:

![Logistic Regression](./0_logisticRegression.png)

#

### **Using Sklearn on Python**

Clone/download this repo, open & run python script: __*3_logReg_plot1.py*__. It will create a plot figure of dataset with its best fit line. Make sure you have installed __*pandas*__, __*numpy*__, __*matplotlib*__ & __*sklearn*__ packages!

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('1_data.csv', sep=',')
# print(df)

# =========================================
# split datasets = data train & data test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    df[['usia']], 
    df['beliAsuransi'], 
    test_size=.1
)

# =========================================
# logistic regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')

# training model
model.fit(x_train, y_train)

# m coef & b intercept
print(model.coef_)
print(model.intercept_)

# accuracy
print(model.score(x_train, y_train))

# =========================================

# prediction
# print(x_test)
# print(model.predict(x_test))
print(model.predict([[40]]))

# probablility
# print(model.predict_proba(x_test))
print(model.predict_proba([[40]]))

# =========================================

# plot dataframe
plt.scatter(df['usia'], df['beliAsuransi'])

dataplot = np.linspace(0,70,150)
def plotlogreg(line):
    return 1 / (1 + np.exp(-line))
bestfit = plotlogreg(
    (model.coef_ * dataplot) + model.intercept_).ravel()

# plot best fit line logistic regression: sigmoid func
plt.plot(
    dataplot,
    bestfit,
    'r-'
)

plt.xlabel('Usia')
plt.ylabel('Membeli Asuransi')
plt.title('Data pengguna asuransi')
plt.show()

```

![Logistic Regression](./3_logReg_plot1.png)

#

### **Logistic Regression on Digits Sklearn Toy Datasets**

Clone/download this repo, open & run python script: __*5_logreg_digit.py*__. It will create a heatmap plot figure of dataset with its prediction, actual value & accuracy score.

```python

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

# prediction
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

```

![Logistic Regression](./5_logreg_digit.png)

#

#### Lintang Wisesa :love_letter: _lintangwisesa@ymail.com_

[Facebook](https://www.facebook.com/lintangbagus) | 
[Twitter](https://twitter.com/Lintang_Wisesa) |
[Google+](https://plus.google.com/u/0/+LintangWisesa1) |
[Youtube](https://www.youtube.com/user/lintangbagus) | 
:octocat: [GitHub](https://github.com/LintangWisesa) |
[Hackster](https://www.hackster.io/lintangwisesa)