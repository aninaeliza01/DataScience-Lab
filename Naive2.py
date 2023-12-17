from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

nav=GaussianNB()
nav.fit(x_train,y_train)
v=nav.predict(x_test)
print("Accuracy on test set:",accuracy_score(y_test,v))

new_data = [[14.21, 20.85, 92.55, 623.9, 0.097, 0.123, 0.089, 0.037, 0.21, 0.056, 0.433, 1.268, 2.844, 32.41, 0.01, 0.014, 0.03, 0.009, 0.025, 0.003, 14.91, 26.5, 98.87, 567.7, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173,]]
new_prediction = nav.predict(new_data)
predicted_category = cancer.target_names[new_prediction[0]]
print("Prediction for new sample:", predicted_category)
