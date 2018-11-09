#Import Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm 
sns.set(style="whitegrid")

df=pd.read_csv("customer_churn.csv")

print (df.head())

print (df.shape)

print (df.info)

#Data is successfully loaded into df pandas dataframe
#a
print (df.dtypes)

#Setting customer_id as index
df.set_index('customer_id')


#Following Statistics summary for three numeric columns
print (df.describe())

#b
#Which attribute has missing values.
df.isnull().values.any
# True shows we have missing values in our dataframe
# Lets find out the column names with missing values

print (df.isnull().sum())

#We have 11 missing values in total_revenue column.

#C
print(df[df["total_revenue"].isnull()]['total_revenue'])

nans = lambda df: df[df.isnull().any(axis=1)]
print (nans(df))
#In all rows where we have null values we can see that the reasons is
    # months colum is 0
    # senior column is no

#d
#How to deal with missing values.
    #We have 7043 records in our dataset and have only 11 missing values which are 0.15 of our dataset.
#it means 6th part of one percent of dataset.
#We have one identity of months column that it is 0 for all rows and senior column is no
#Rest of the columns have mixed values.
    #We should remove the missing values so it has no effect on our dataset.
#We can use our new dataset for further processing
    #Removing missing values
df.dropna(axis=0,inplace=True)

#Can verify we have no missing values now
print (df.isnull().sum())

#e
#Discover inconsistencies
print (df.head(5))

#Our dataset is normaly distributed and no inconsistency can be seen. 

#Task 2
# Visulization
#a
df1=df
print (df1.monthly_fee.unique())


df.churn.unique()
df.churn=df.churn.replace('no',0)
df.churn=df.churn.replace('yes',1)
df.churn.unique()
print (df.dtypes)

df['churn'].groupby(df['months']).sum().plot.bar()

df['churn'].groupby(df['monthly_fee']).sum().plot.bar()

df['churn'].groupby(df['total_revenue']).sum().plot.bar()

# Create a new temporary dataframe to help us plot these variables.
df1 = pd.melt(df, id_vars=['Churn'], value_vars=["months"], var_name='variable' )

# Create a factorplot
g = sns.factorplot( x="variable", y="Churn", hue='value', data=df1, size=4, aspect=2, kind="bar", palette="husl", ci=None )
g.despine(left=True)
g.set_ylabels("Churn Rate")
plt.show()


# Create a new temporary dataframe to help us plot these variables.
df2 = pd.melt(df, id_vars=['Churn'], value_vars=["monthly_fee"], var_name='variable' )

# Create a factorplot
g = sns.factorplot( x="variable", y="Churn", hue='value', data=df2, size=4, aspect=2, kind="bar", palette="husl", ci=None )
g.despine(left=True)
g.set_ylabels("Churn Rate")
plt.show()

#I have displayed churn rate for two columns. for the sack of better understanding and shorten code.
# You can change the columns names in value_vars and get for rest of the coumns.
#B
#Will be done by same code.
#c
#To check how the customer has developed in last 12 months

#To check this we needs data or time for customer segment on monthly basis.
#no column or fields are related to help this question.
#i am attaching the code for make sure how the customer can be categorized on monthly basis.

print (dt.datetime.now())
#df.resample('M').sum()
#We have dattime object , by using this we can resample dataframe on minutes, days and monthly basis. 
#but we dont have time in our scenerio.

#Task 3
#Clustering 

#get values and plot
f1=df['months'].values
f2=df['monthly_fee'].values
f3=df['gender'].values
f4=df['partner'].values
f5=df['family'].values
f6=df['senior'].values
f7=df['mobile'].values
f8=df['dual_sim'].values
f9=df['device_insurance'].values
f10=df['internet'].values
f11=df['web_security'].values
f12=df['cloud'].values
f13=df['support'].values
f14=df['tv_replay'].values
f15=df['subscription'].values
f16=df['paperless_invoice'].values
f17=df['total_revenue'].values

X= np.array(list(zip(f1,f2)))

#Eculidean distance calculater
def dist(a,b, ax=1):
	return np.linalg.norm(a-b, axis=ax)

#No of Clusters
k=3
#x corrodinates of centroids
C_x=np.random.randint(0, np.max(X)-20, size=k)
#x corrodinates of centroids
C_y=np.random.randint(0, np.max(X)-20, size=k)
C=np.array(list(zip(C_x, C_y)), dtype= np.float32)
print('-------------')
print(C)



#plotting alongside with Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker= '*', s=200, c='g')
plt.show()




# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)



colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
plt.show()

#K-means model
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
# Comparing with scikit-learn centroids
print(C) # From Scratch
print(centroids) # From sci-kit learn


# Task 4

# Classification

print (df.columns)
logreg=LogisticRegression()
#Updating the feature list
cols=["gender", "partner", "family", "senior", "mobile", "dual_sim", "device_insurance", 
      "internet", "web_security", "cloud", "support", "tv_replay", "video_on_demand", "subscription", "paperless_invoice", 
      "payment_method", "monthly_fee", "months","total_revenue"] 
X= df[['months','monthly_fee','total_revenue']]
y=df['churn']

#Model
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)




