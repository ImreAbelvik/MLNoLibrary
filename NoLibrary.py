# Importing librarys
import matplotlib.pyplot as plt
import pandas as pd
import math

# Creating functions
# function to calculate the coefficients and intercept for y_hat
def linear_regression_coefficients(x, y):
    # Calculate the mean for list x and list y
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(x)

    # Calculate the numerator and denominator for O1
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    # Calculate O1 and O0 coefficients
    O1 = numerator / denominator
    O0 = y_mean - O1 * x_mean

    return O1, O0

# function to generate the y values for x in data sett to create best fitt line
def generate_y_list(x, O1, O0):
    return [int(O0 + O1 * xi) for xi in x]

def calcualte_error(y, y_hat):
    # Calcualte Mean Square Error
    MSE = sum(((yi - y_hat_i) ** 2) for yi, y_hat_i in zip(y, y_hat)) / len(y_hat)

    # Calculate Mean Absulute Error
    MAE = 1 / len(y_hat) * sum (yi - y_hat_i for yi, y_hat_i in zip(y, y_hat))

    # Calcualte Root Mean Square Error
    RMSE = math.sqrt(MSE)

    return f"""
    Mean Square Error: { int(MSE) }
    Mean Absulute Error: { MAE }
    Root Mean Square Error: { int(RMSE) }
    """

# Reading the data
df = pd.read_csv("Salary_Data.csv")

# Prepering the data
df["Salary"] = df["Salary"].astype(int)

# Using function linear_regression_coefficients to get the coefficients for dataset 
O1, O0 = linear_regression_coefficients(df["YearsExperience"], df["Salary"])

# Getting a list for he Y_hat
y_hat_list = generate_y_list(df["YearsExperience"], O1, O0)

# Creating the title and labels for the canvas
plt.title("Salary for years of experiense")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")


plt.scatter(df["YearsExperience"], df["Salary"])
plt.plot(df["YearsExperience"], y_hat_list)
plt.show()

# Using the function calculate error to calculate the MSE, MAE, RMSE
print(calcualte_error(df["Salary"], y_hat_list))
