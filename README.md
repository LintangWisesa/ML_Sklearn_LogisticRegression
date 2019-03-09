![simplinnovation](https://4.bp.blogspot.com/-f7YxPyqHAzY/WJ6VnkvE0SI/AAAAAAAADTQ/0tDQPTrVrtMAFT-q-1-3ktUQT5Il9FGdQCLcB/s350/simpLINnovation1a.png)

# Sklearn: Logistic Regression

### **Basic Formula**

In statistics, __logistic regression__ is a predictive analysis that used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. Here is the basic formula of logistic regression:

![Rumus Logistic Regression](data:image/png;base64,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)

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

#### Lintang Wisesa :love_letter: _lintangwisesa@ymail.com_

[Facebook](https://www.facebook.com/lintangbagus) | 
[Twitter](https://twitter.com/Lintang_Wisesa) |
[Google+](https://plus.google.com/u/0/+LintangWisesa1) |
[Youtube](https://www.youtube.com/user/lintangbagus) | 
:octocat: [GitHub](https://github.com/LintangWisesa) |
[Hackster](https://www.hackster.io/lintangwisesa)