from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sea
import matplotlib.pyplot as py

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

data = fetch_20newsgroups(subset="train", shuffle=True, random_state=42, categories=categories)

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(data.data)
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

svm = SVC(kernel='linear', random_state=42)
svm.fit(x_train, y_train)

predictions = svm.predict(x_test)
print("Accuracy:", accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions, target_names=data.target_names))

new_data = [
    "I have a question about computer graphics",
    "This is a medical-related report",
]

x_new = vectorizer.transform(new_data)
pred = svm.predict(x_new)

for i, text in enumerate(new_data):
    print(f"{text}")
    print(data.target_names[pred[i]])


confusion_matrix=confusion_matrix(y_test,predictions)
print(confusion_matrix)
sea.heatmap(confusion_matrix,annot=True,fmt='g')
py.show()
