import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
dataDg = load_digits()

# split dataset
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    dataDg['data'],
    dataDg['target'],
    test_size = .1
)

# logistic regression
model = LogisticRegression(
    solver='lbfgs', 
    multi_class='auto',
    max_iter=1000
)
model.fit(xtr, ytr)

# predict a picture
from PIL import Image
import PIL.ImageOps

gbr = Image.open('8_gbr2.jpg').convert('L')
gbr = gbr.resize((8, 8))
gbr = PIL.ImageOps.invert(gbr)
gbrArr = np.array(gbr)
gbrArr2 = gbrArr.reshape(1, 64)
prediksi = model.predict(gbrArr2.reshape(1, -1))
print(prediksi[0])

plt.imshow(gbrArr, cmap='gray')
plt.title('Prediksi : {}'.format(prediksi[0]))
plt.show()
