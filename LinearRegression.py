from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as py

data=pd.read_csv("Salary_Data.csv")
x=data['YearsExperience'].values.reshape(-1,1)
y=data['Salary'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

linear=LinearRegression()
linear.fit(x_train,y_train)
v=linear.predict(x_test)
r2 = r2_score(y_test, v)
print("R-squared:", r2)

py.scatter(x_test,y_test)
py.plot(x_test,v)
py.show()