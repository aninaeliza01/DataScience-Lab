from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

digit=load_digits()
x=digit.data
y=digit.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
v=clf.predict(x_test)
test_accuracy=accuracy_score(y_test,v)
print("Accuracy on test set:", test_accuracy)

