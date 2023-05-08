#Importing library
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression



# Data Preparation

churndata = pd.read_csv('AIA_Churn_Modelling_Case_Study.csv')
churndata.info()
churndata.describe()

churndata['TotalCharges']=pd.to_numeric(churndata['TotalCharges'], errors='coerce')
churndata.isnull().sum()  #check Null
churndata.dropna(inplace=True) # because NaN accounts for small proportion of dataset, so I remove it instead of replacing it with mean or median, ...
churndata.drop(columns='customerID', inplace=True)


# DESCRIPTIVE ANALYTICS
#percentage of target values
sns.countplot(x = "Churn", data = churndata )       
churndata.loc[:, 'Churn'].value_counts()


#Categorial columns
categorial_data=churndata.drop(columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
for col in categorial_data.columns:
    pd.crosstab(churndata[col], churndata['Churn']).plot(kind='bar', stacked=True, rot=0)
# Numerical columns
numerical_data=churndata[['tenure', 'MonthlyCharges', 'TotalCharges']]
for col in numerical_data.columns:
    sns.histplot(x=churndata[col], hue = churndata['Churn'], multiple = 'stack')
    
 
#ENCODING
#BinaryClass
label_col = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']
for col in label_col:
    if col == 'gender':
        churndata[col] = churndata[col].map({'Female': 1, 'Male': 0})
    else: 
        churndata[col] = churndata[col].map({'Yes': 1, 'No': 0})    
#MultiClass
onehot_col = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']
churndata = pd.get_dummies(churndata, columns = onehot_col, drop_first=True, dtype=int) 

#NORMALIZATION
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
churndata['tenure'] = sc.fit_transform(churndata[['tenure']])
churndata['MonthlyCharges'] = sc.fit_transform(churndata[['MonthlyCharges']])
churndata['TotalCharges'] = sc.fit_transform(churndata[['TotalCharges']])

#CORRELATION
corr = churndata.corr()
churndata.corr()['Churn'].plot(kind='bar', figsize=(12,5))


# SPLIT DATA
X=churndata.drop(columns='Churn')
Y=churndata['Churn']
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state =11 ,stratify=Y, test_size = 0.2)

#FEATURE IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features Importance')
    plt.tight_layout()
    plt.show()
plot_importance(GradientBoostingClassifier(random_state=11).fit(x_train, y_train), x_train)

#MODELS
#KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 30)
kn.fit(x_train,y_train)
y_pred1 = kn.predict(x_test)
confusion_matrix(y_test,y_pred1)
accuracy_score(y_test, y_pred1)

#LogisticRegression
log=LogisticRegression(random_state=11)
log.fit(x_train, y_train)
y_pred2=log.predict(x_test)
confusion_matrix(y_test,y_pred2)
accuracy_score(y_test, y_pred2)

#Support vector classifier
svc = SVC(random_state=11)
svc.fit(x_train, y_train)
y_pred3 = svc.predict(x_test)
confusion_matrix(y_test,y_pred3)
accuracy_score(y_test, y_pred3)

#RandomForestClassifier
randomforest=RandomForestClassifier(random_state=11)
randomforest.fit(x_train, y_train)
y_pred4 = randomforest.predict(x_test)
confusion_matrix(y_test, y_pred4)
accuracy_score(y_test, y_pred4)

#GradientBoostingClassifier
Gra=GradientBoostingClassifier(random_state=11)
Gra.fit(x_train, y_train)
y_pred5 = Gra.predict(x_test)
confusion_matrix(y_test, y_pred5)
accuracy_score(y_test, y_pred5)


#HYPERPARAMETER TUNING
param_grid = [{'C': list(np.arange(0.1,10,0.1))}]
grid_search = GridSearchCV(log, param_grid, cv=5, verbose=1, return_train_score = True).fit(x_train, y_train)
best_param=grid_search.best_params_
print(best_param)

log=LogisticRegression(C=best_param)
log.fit(x_train, y_train)
y_pred=log.predict(x_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test, y_pred)
