from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

tree = DecisionTreeClassifier(max_depth=3)
tree.fit(x_train, y_train)

y_pred = tree.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_data = [[5.1, 3.5, 1.4, 0.2]]

predicted_class = tree.predict(new_data)
print("Predicted class for new data:", iris.target_names[predicted_class])

plt.figure(figsize=(10, 8))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
