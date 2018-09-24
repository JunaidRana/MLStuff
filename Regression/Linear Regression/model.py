import matplotlib
#matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd 

#Load Csv
df=pd.read_csv("Housing.csv")
print df.head()

print ("-------------")

Y=df['price']
X=df['lotsize']

X=X.reshape(len(X),1)
Y=Y.reshape(len(Y),1)

X_train= X[:-250]
X_test=X[-250:]

Y_train= Y[:-250]
Y_test=Y[-250:]

plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

regr= linear_model.LinearRegression()
regr.fit(X_train, Y_train)
plt.plot(X_test,regr.predict(X_test),color='red', linewidth=3)

plt.show()

print (str(round (regr.predict(5000))))

