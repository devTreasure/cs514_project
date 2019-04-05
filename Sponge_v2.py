from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

data= pd.read_csv('H:\\unsupervised_learning\\sponge.data')

print('---Data Describe---')
data.describe()
print('---Data Describe---')
#X.drop(labels=(['Channel', 'Region']), axis=1, inplace=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
#print(X)

true_k =3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=1)
model.fit(X)
predicted = model.predict(X)
print('###############################')
print(predicted)
print('###############################')
order_centroids = model.cluster_centers_.argsort()
terms = vectorizer.get_feature_names()
print('#############Centroid##################')
print(order_centroids)
print('#############Centroid##################')
#print(model.cluster_centers_)

for i in range(true_k):
 print("Cluster %d:" % i)
for ind in order_centroids[i]:
 print("%s"  % terms[ind])