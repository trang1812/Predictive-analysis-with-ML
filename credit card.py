# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:22:32 2023

@author: Win 10
"""

#import library
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
%matplotlib inline
pd.options.display.float_format = '{:.2f}'.format


import pandas as pd
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from collections import Counter #this to quickly assess class split in y

#import data
data = pd.read_csv(r'C:\Users\Win 10\Downloads\Projects\Predictive analytics with ML\credit card default\creditcard.csv')
data.head()

#check data
data.info()
data.describe()

#check missing data
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

#check data balancing
LABELS = ["Not Fraud", "Fraud"]
count_classes = pd.value_counts(data['Class'], sort = True)
(count_classes/data.shape[0]).plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Percentage");

#EDA
#group data by time and class
data['Hour'] = data['Time'].apply(lambda x: np.floor(x / 3600))
tmp = data.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df_time = pd.DataFrame(tmp)
df_time.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']

#Amount by time
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df_time.loc[df_time.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df_time.loc[df_time.Class==1], color="red")
plt.suptitle("Total Amount")
plt.show()

#Number of transaction by time
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df_time.loc[df_time.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df_time.loc[df_time.Class==1], color="red")
plt.suptitle("Total Number of Transactions")
plt.show()

#Outlier
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data, palette="PRGn",showfliers=False)
plt.show();

#Feature correlation
plt.figure(figsize = (14,14))
plt.title('Features correlation plot')
corr = data.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()

#Features density plot
features = data.columns.values
class_0 = data.loc[data['Class'] == 0]
class_1= data.loc[data['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

i = 0
for feature in features:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(class_0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(class_1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show() 

#data spliting
X = data.copy(deep=True)
y = X.pop('Class')

#split training and test data. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=2017, test_size = .25)

#apply random downsampling to balance the dataset
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

#what's the balance now?
print('Original dataset shape %s' % Counter(y_resampled))

#Now run auto ML on the balanced, smaller dataset
clf = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs = -1)
clf.fit(X_resampled, y_resampled)
#what pipeline did tpot choose?
print(clf.fitted_pipeline_)

#evaluate the confusion matrix
confusion_matrix(y_train, clf.predict(X_train))

#Apply random downsampling to balance the dataset
rus = RandomUnderSampler(sampling_strategy = .25)
X_resampled2, y_resampled2 = rus.fit_resample(X_train, y_train)
#check data balance
print('Original dataset shape %s' % Counter(y_resampled2))

clf2 = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs = -1)
clf2.fit(X_resampled2, y_resampled2)
confusion_matrix(y_train, clf2.predict(X_train))

#SMOTE
X_resampled3, y_resampled3 = SMOTE().fit_resample(X_resampled2, y_resampled2)
print('Original dataset shape %s' % Counter(y_resampled3))
clf3 = TPOTClassifier(generations=8, population_size=65, verbosity=2, n_jobs = -1)
clf3.fit(X_resampled3, y_resampled3)
confusion_matrix(y_train, clf3.predict(X_train))

#Apply random downsampling to balance the dataset, but downsample less than before
rus = RandomUnderSampler(sampling_strategy = .125)
X_resampled4, y_resampled4 = rus.fit_resample(X_train, y_train)
#what's the balance now?
print('Original dataset shape %s' % Counter(y_resampled4))
#let's re-run tpot with these data, using the same tpot parameters as in clf2
clf4 = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs = -1)
clf4.fit(X_resampled4, y_resampled4)
#and evaluate the confusion matrix
confusion_matrix(y_train, clf4.predict(X_train))

#Machine learning models
#Random Forest
clf = RandomForestClassifier(n_jobs=4, random_state=2023, n_estimators=100, verbose=False)
clf.fit(X_resampled4, y_resampled4)

confusion_matrix(y_test, clf.predict(X_test))
accuracy_score(y_test,clf.predict(X_test))

#Adaboost
clf2 = AdaBoostClassifier(random_state=2023, learning_rate=0.5,n_estimators=100)
clf2.fit(X_resampled4, y_resampled4)

confusion_matrix(y_test, clf2.predict(X_test))
accuracy_score(y_test,clf2.predict(X_test))

#Logistic Regression
clf3 = LogisticRegression()
clf3.fit(X_resampled4, y_resampled4)
confusion_matrix(y_test, clf3.predict(X_test))
accuracy_score(y_test,clf3.predict(X_test))
