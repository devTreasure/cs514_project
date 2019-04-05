from  sklearn.cluster import k_means
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
X= pd.read_csv('H:\\unsupervised_learning\\sponge.data')
#X.drop(labels=(['Channel', 'Region']), axis=1, inplace=True)
vectorizer.fit(X)
print('sponge dataset has {} samples with {} features each'.format(*X.shape))
print(X.shape)
print(X.head())
#km = k_means(X=X_data,n_clusters=3)

#print(X['DURO']) /# some fatures



#model = GaussianMixture(n_components=3, covariance_type="full")
#model.fit(X_train)
#log_prob = model.score_samples(X_train)
#outliers = get_outliers(log_prob, 0.15)
#data["Outlier"] = outliers
#plt.scatter(X[:,0],X[:,1], label='True Position')
#plt.interactive(False)
#plt.show()
#y=pd.get_dummies(X)
#X=np.asfarray(X,float)
#print(y)
#kmeans =  k_means(X,n_clusters=4)


#kmeans.fit(X)
#y_kmeans = kmeans.predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#from sklearn.manifold import TSNE

# Project the data: this step will take several seconds
#tsne = TSNE(n_components=2, init='random', random_state=0)
#digits_proj = tsne.fit_transform(X)

# Compute the clusters
#kmeans = k_means(n_clusters=10, random_state=0)
#clusters = kmeans.fit_predict(digits_proj)

#