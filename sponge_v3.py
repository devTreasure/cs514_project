from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data= pd.read_csv('H:\\unsupervised_learning\\sponge.data')
data = np.array(data)
print(data.shape)
data = pd.to_numeric(data)
#data = data[:,1:46]
#num_data = pd.get_dummies(data)
#print(num_data.shape)
#print(num_data[0])
print(data[0])

#data= data.flatten()
#print(data)
#print(data[0])
#df_with_dummies = pd.get_dummies(data)
#print(df_with_dummies[0])

from sklearn.manifold import TSNE

# Project the data: this step will take several seconds
tsne = TSNE(n_components=2, init='random', random_state=0)
digits_proj = tsne.fit_transform(data)

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

#accuracy_score(digits.target, labels)


from sklearn import preprocessing

# create the Labelencoder object
le = preprocessing.LabelEncoder()