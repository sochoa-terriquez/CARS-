import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Import the data
cars = pd.read_csv("CARS.csv")

print("First 5 rows of the dataset:")
print(cars.head())

print("\nColumn names:")
print(cars.columns)

# 2. Count how many cars for each Origin
origin_counts = cars.groupby("Origin").size()

print("\nNumber of cars by Origin:")
print(origin_counts)

# 3. Create bar chart for Origin counts
plt.figure(figsize=(8, 6))
ax = origin_counts.plot(kind="bar")

for i, value in enumerate(origin_counts):
    ax.text(i, value, str(value), ha="center", va="bottom")

plt.xlabel("Origin")
plt.ylabel("Number of Cars")
plt.title("Number of Cars by Origin")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Find missing values in each column
missing_values = cars.isnull().sum()

print("\nMissing values in each column:")
print(missing_values)

# 5. One-Way ANOVA: MPG_City across 4, 6, and 8 cylinders
mpg_4 = cars[cars["Cylinders"] == 4]["MPG_City"].dropna()
mpg_6 = cars[cars["Cylinders"] == 6]["MPG_City"].dropna()
mpg_8 = cars[cars["Cylinders"] == 8]["MPG_City"].dropna()

f_stat, p_value = f_oneway(mpg_4, mpg_6, mpg_8)

print("\nOne-Way ANOVA Results:")
print("F-statistic:", f_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Conclusion: Reject the null hypothesis.")
    print("There is a significant difference in average MPG_City among 4, 6, and 8 cylinder cars.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("There is no significant difference in average MPG_City among 4, 6, and 8 cylinder cars.")

# 6. Multiple Linear Regression: Predict MPG_City using Horsepower and Weight
regression_data = cars[["MPG_City", "Horsepower", "Weight"]].dropna()

X = regression_data[["Horsepower", "Weight"]]
y = regression_data["MPG_City"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\nMultiple Linear Regression Results:")
print(model.summary())

# Error metrics
predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)

print("\nRegression Error Metrics:")
print("R-squared:", model.rsquared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
