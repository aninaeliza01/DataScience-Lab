from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

test_predictions = clf.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy on test set:", test_accuracy)

new_data = [[5.1, 3.5, 1.4, 0.2]]
new_prediction = clf.predict(new_data)
predicted_category = iris.target_names[new_prediction[0]]
print("Prediction for new sample:", predicted_category)


