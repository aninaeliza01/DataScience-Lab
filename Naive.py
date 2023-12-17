from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

nav=GaussianNB()
nav.fit(x_train,y_train)
v=nav.predict(x_test)
print("Accuracy on test set:",accuracy_score(y_test,v))

new_data = [[5.1, 3.5, 1.4, 0.2]]
new_prediction = nav.predict(new_data)
predicted_category = iris.target_names[new_prediction[0]]
print("Prediction for new sample:", predicted_category)