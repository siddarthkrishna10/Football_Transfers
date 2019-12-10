import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Reading the dataset into an object
d = pd.read_csv('https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20I/Transfers.csv')

#Cleaning the Dataset of NaN values
d1 = d.dropna()

#Plotting the data in pairplots using seaborn
pp = sns.pairplot(d1, y_vars=['Age'], x_vars=['Market_value', 'Transfer_fee'], hue="Season")
plt.show()
