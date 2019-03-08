import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('1_data.csv', sep=',')
print(df)

# plot dataframe
plt.scatter(df['usia'], df['beliAsuransi'])
plt.xlabel('Usia')
plt.ylabel('Membeli Asuransi')
plt.title('Data pengguna asuransi')
plt.show()