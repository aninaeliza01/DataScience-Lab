from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as py

iris=load_iris()
x=iris.data
k=KMeans(n_clusters=3,random_state=42)
k.fit(x)
cc=k.labels_
print(k.labels_)
cew=k.cluster_centers_
print(k.cluster_centers_)
py.scatter(x[:,0],x[:,1],c=cc)
py.scatter(cew[:,0],cew[:,1],marker="*")
py.show()

