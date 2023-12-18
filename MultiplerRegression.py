from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

data = fetch_california_housing()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

l = LinearRegression()
l.fit(x_train, y_train)

v = l.predict(x_test)

r2 = r2_score(y_test, v)
mse = mean_squared_error(y_test, v)

print("R-squared:", r2)
print("Mean Squared Error:", mse)

result_df = pd.DataFrame({"Actual": y_test, "Predicted": v})
print(result_df)

print("Coefficients:", l.coef_)
print("Intercept:", l.intercept_)
