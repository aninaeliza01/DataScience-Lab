import numpy as np
from sklearn.linear_model import LinearRegression
y= np.array([55, 60, 65, 70, 80]).reshape(-1, 1)
x = np.array([52, 54, 56, 58, 62])
model = LinearRegression()
model.fit(y,x)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
