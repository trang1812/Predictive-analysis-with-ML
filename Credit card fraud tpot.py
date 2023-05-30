###################
#This script runs tpot (an automated ML library) on the European credit card fraud data.
#   The script illustrates how tpot searches, how computationally intensive auto-ml is,
#   and how to use resampling in conjunction with auto-ml.
#data source: https://www.kaggle.com/mlg-ulb/creditcardfraud
###################

#Import some libraries
import pandas as pd
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from collections import Counter #this to quickly assess class split in y

#Get credit card defaults data
df = pd.read_csv('creditcard.csv')
X = df.copy(deep=True)

#isolate data from target
y = X.pop('Class')

#what was the orignal balance of y?
print('Original dataset shape %s' % Counter(y))

#split training and test data. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=2017, test_size = .25, stratify = y)

#splitting has preserved the class balance
print('Original dataset shape %s' % Counter(y_train))

##################
#round-1 auto-ml via tpot on these data
##################

#Instead of using all data, apply random downsampling to balance the dataset
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

################
#Let's see what happens if we instead use more of the majority class
################

#Apply random downsampling to balance the dataset
rus = RandomUnderSampler(sampling_strategy = .25)
X_resampled2, y_resampled2 = rus.fit_resample(X_train, y_train)
#what's the balance now?
print('Original dataset shape %s' % Counter(y_resampled2))

#re-run tpot with these data
clf2 = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs = -1)
clf2.fit(X_resampled2, y_resampled2)
#tpot seems to be doing better (based on the 'best internal CV score'), but also seems to stagnate
#evaluate the confusion matrix
confusion_matrix(y_train, clf2.predict(X_train))

##################
# throw in some synthetic observations using SMOTE
X_resampled3, y_resampled3 = SMOTE().fit_resample(X_resampled2, y_resampled2)
#what's the balance now?
print('Original dataset shape %s' % Counter(y_resampled3))

#re-run tpot with these data
clf3 = TPOTClassifier(generations=8, population_size=65, verbosity=2, n_jobs = -1)
clf3.fit(X_resampled3, y_resampled3)
#and evaluate the confusion matrix
confusion_matrix(y_train, clf3.predict(X_train))


##############
#Perhaps try making the imbalance greater, to keep more of the negative class (since we have a FP problem)?
##############

#Apply random downsampling to balance the dataset, but downsample less than before
rus = RandomUnderSampler(sampling_strategy = .125)
X_resampled4, y_resampled4 = rus.fit_resample(X_train, y_train)
#what's the balance now?
print('Original dataset shape %s' % Counter(y_resampled4))

#re-run tpot with these data, using the same tpot parameters as in clf2
clf4 = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs = -1)
clf4.fit(X_resampled4, y_resampled4)
#and evaluate the confusion matrix
confusion_matrix(y_train, clf4.predict(X_train))

