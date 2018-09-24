import matplotlib
#matplotlib.use('GTKAgg')
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import numpy as np
from sklearn import datasets, metrics, linear_model , preprocessing
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



#Load Csv
df=pd.read_csv("banking.csv")
print (df.shape)
print df.head()

print (df['education'].unique())

#make all education basic to one sinle 'Basic'
df['education']=np.where(df['education']=='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education']=='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education']=='basic.4y', 'Basic', df['education'])

print (df['education'].unique())

print (df['y'].value_counts())

sns.countplot(x='y', data=df, palette= 'hls')
plt.show()
plt.savefig('count_plot')


print (df.groupby('y').mean())
print ('-----------------------------')

print(df.groupby('job').mean())
print ('-----------------------------')

print (df.groupby('marital').mean())
print ('-----------------------------')

print(df.groupby('education').mean())
print ('-----------------------------')


#%matplotlib inline
# pd.crosstab(df.job,df.y).plot(kind='bar')
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('purchase_fre_job')




# table=pd.crosstab(df.marital,df.y)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Marital Status vs Purchase')
# plt.xlabel('Marital Status')
# plt.ylabel('Proportion of Customers')
# plt.savefig('mariral_vs_pur_stack')




# table=pd.crosstab(df.education,df.y)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Education vs Purchase')
# plt.xlabel('Education')
# plt.ylabel('Proportion of Customers')
# plt.savefig('edu_vs_pur_stack')


# pd.crosstab(df.day_of_week,df.y).plot(kind='bar')
# plt.title('Purchase Frequency for Day of Week')
# plt.xlabel('Day of Week')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('pur_dayofweek_bar')



# pd.crosstab(df.month,df.y).plot(kind='bar')
# plt.title('Purchase Frequency for Month')
# plt.xlabel('Month')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('pur_fre_month_bar')



# df.age.hist()
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.savefig('hist_age')


# pd.crosstab(df.poutcome,df.y).plot(kind='bar')
# plt.title('Purchase Frequency for Poutcome')
# plt.xlabel('Poutcome')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('pur_fre_pout_bar')



cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    data1=df.join(cat_list)
    df=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


data_final=df[to_keep]
print ('final values')
print (data_final.columns.values)

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]


#Recursive Feature Elimination (RFE)
logreg=LogisticRegression()

rfe=RFE(logreg,18)
rfe= rfe.fit(data_final[X], data_final[y])
print (rfe.support_)
print (rfe.ranking_)

#Updating the feature list
cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
X=data_final[cols]
y=data_final['y']


#Model
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


#10-fold Cross validation
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))




#Correct and incorrect predictions
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)



#Precison (true postive or false positive) beta
print(classification_report(y_test, y_pred))


#ROC Curve to show prediction accuracy
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
