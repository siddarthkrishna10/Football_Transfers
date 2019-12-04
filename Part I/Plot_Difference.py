import pandas as pd
import matplotlib.pyplot as plt

#Reading the data from Mean_Table into an object
b = pd.read_csv('C:/Users/Siddhardh/Desktop/OiDS Project/Code/Mean_Table.csv')

#Plotting a line graph for the data
b.plot(x='Season', y='Average Difference')
plt.title('The Average Difference Over All Seasons')
plt.xlabel('Season')
plt.ylabel('Average Difference')
plt.show()