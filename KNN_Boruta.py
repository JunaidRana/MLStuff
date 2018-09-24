import pandas as pd
import numpy as np

#This dataset is organised by feature importance.
df = pd.read_csv('normcleve_knn.csv')
df1 = df
array = df1.values
#We already know the feature importance and ranking through Boruta files.
#Here we are taking only two columns and our accuracy rate is still 96.7 %
X = array[:,0:2]
y = array[:,14]
y = y.astype(int)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print (y_pred)
print (knn.score(X_test, y_test))
