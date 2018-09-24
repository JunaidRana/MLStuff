from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import tree


#Classfifer
X=[[0,0] , [1,1]]
Y=[0,1]
clf=DecisionTreeClassifier()
clf=clf.fit(X,Y)
print (clf)

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)



#Regressor

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])

