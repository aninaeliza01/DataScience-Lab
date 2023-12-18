from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as py

cancer=load_breast_cancer()
x=cancer.data
kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(x)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
cen=kmeans.cluster_centers_
py.scatter(x[:,0],x[:,1],c=kmeans.labels_)
py.scatter(cen[:,0],cen[:,1],marker="*",color="red")
py.show()