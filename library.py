# Importing librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reding the csv file to a pandas data frame
df = pd.read_csv("Salary_Data.csv")

# Asiging the X and y value, as a numpy array with shape (x, 1)
X = np.array(df["YearsExperience"]).reshape(-1,1)
y = np.array(df["Salary"]).reshape(-1,1)

# Using the train test split function to devide the data betwin the training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=None)

# Chosing the ML model
from sklearn import linear_model
regr = linear_model.LinearRegression()

# Traing the model on the data
regr.fit(X_train, y_train)

# Predictig Y values for the X test valus using the model
y_pred = regr.predict(X_test)

# Creating a canvas, and Scatering the traing data with the Y_hat line on the canvas
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred)
plt.show()

# Creating canvas 2, and Scatering the test data on the canvas with line for the Y_hat
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.show()

# Calculatiing the error of the ML
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = r2_score(y_test, y_pred)

print(f"""
Mean Square Error: { MSE }
Mean Absulute Error: { MAE }
Root Mean Square Error: { RMSE }
""")