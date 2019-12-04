import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

c = pd.read_csv('C:/Users/Siddhardh/Desktop/OiDS Project/Code/Transfers.csv')

c1 = c.dropna()

#Creating your Linear Regression objects; the dependant variable and explanatory variable
X = c1['Transfer_fee'].values.reshape(-1, 1)
y = c1['Market_value'].values.reshape(-1, 1)

#Splitting your data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Using Linear Regression from scikit learn, fitting the model and calculating the intercepts and slope
cx = LinearRegression()
cx.fit(X_train, y_train)
print('Intercept:', cx.intercept_)
print('Coefficient:', cx.coef_)

#Predicting using test data
y_pred = cx.predict(X_test)

#Writing the actual and predicted values into a csv file
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.to_csv('C:/Users/Siddhardh/Desktop/OiDS Project/Code/Actual_Predicted.csv')

#Plotting the scatter plot and linear regression line
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=1)
plt.show()

mean = c1["Market_value"].mean()
print('Mean of Market Value:', mean)

#Calclulating evaluation metrics to evalute your regression model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))