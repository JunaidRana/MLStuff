import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
df = pd.read_csv('normcleve.csv')
df1 = df
# print (df1.head(20))
# print (df1.info())
array = df1.values
X = array[:,0:14]
print (X)
y = array[:,14]
y = y.astype(int)
print (y)

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)


# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

# check selected features - first 5 features are selected
a = feat_selector.support_
print (a)
# np.savetxt("Support.csv", a, delimiter=",")

# check ranking of features
b = feat_selector.ranking_
print ("Printing Column Rankings")
print (b)
# np.savetxt("Ranking.csv", b, delimiter=",")

feat_selector.n_features_
# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
# print (X_filtered)
# np.savetxt("Boruta.csv", X_filtered, delimiter=",")
print ("ok")