#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 04:32:30 2019

@author: yudi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv('winequality-red.csv')
df.head()
np.unique(df.quality)
df.info()

#data visualization

#barplot
#relationship between 'fixed acidity'&'quality'
plt.figure(figsize=(20,10))

plt.subplot(3,4,1)
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)

#relationship between 'volatile acidity'&'quality'
plt.subplot(3,4,2)
sns.barplot(x = 'quality', y = 'volatile acidity', data = df)

#relationship between 'citric acid'&'quality'
plt.subplot(3,4,3)
sns.barplot(x = 'quality', y = 'citric acid', data = df)

#relationship between 'residual sugar'&'quality'
plt.subplot(3,4,4)
sns.barplot(x = 'quality', y = 'residual sugar', data = df)

#relationship between 'chlorides'&'quality'
plt.subplot(3,4,5)
sns.barplot(x = 'quality', y = 'chlorides', data = df)

#relationship between 'free sulfur dioxide'&'quality'
plt.subplot(3,4,6)
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = df)

subplot(3,4,7)
plt.sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df)

#relationship between 'sulphates'&'quality'
plt.subplot(3,4,8)
sns.barplot(x = 'quality', y = 'sulphates', data = df)

#relationship between 'alcohol'&'quality'
plt.subplot(3,4,9)
sns.barplot(x = 'quality', y = 'alcohol', data = df)

#relationship between 'density'&'quality'
plt.subplot(3,4,10)
sns.barplot(x = 'quality', y = 'density', data = df)

#relationship between 'pH'&'quality'
plt.subplot(3,4,11)
sns.barplot(x = 'quality', y = 'pH', data = df)

#countplot -- viewing distribution
plt.figure()
sns.countplot(x = 'quality', data = df)

#boxplot -- check outliers
plt.figure(figsize=(20,10))
plt.subplot(3,4,1)
sns.boxplot(x = 'quality', y = 'fixed acidity', data = df)

plt.subplot(3,4,2)
sns.boxplot(x = 'quality', y = 'volatile acidity', data = df)

plt.subplot(3,4,3)
sns.boxplot(x = 'quality', y = 'citric acid', data = df)

plt.subplot(3,4,4)
sns.boxplot(x = 'quality', y = 'residual sugar', data = df)

plt.subplot(3,4,5)
sns.boxplot(x = 'quality', y = 'chlorides', data = df)

plt.subplot(3,4,6)
sns.boxplot(x = 'quality', y = 'free sulfur dioxide', data = df)

plt.subplot(3,4,7)
sns.boxplot(x = 'quality', y = 'total sulfur dioxide', data = df)

plt.subplot(3,4,8)
sns.boxplot(x = 'quality', y = 'sulphates', data = df)

plt.subplot(3,4,9)
sns.boxplot(x = 'quality', y = 'alcohol', data = df)

plt.subplot(3,4,10)
sns.boxplot(x = 'quality', y = 'density', data = df)

plt.subplot(3,4,11)
sns.boxplot(x = 'quality', y = 'pH', data = df)

df.describe()

#data preprocessing

#binary classification
high_quality = []
for i in df['quality']:
    if i >= 1 and i < 6.5:
        high_quality.append('0')
    elif i >=6.5:
        high_quality.append('1')
df['high_quality'] = high_quality
df['high_quality'].value_counts()
plt.figure()
sns.countplot(df['high_quality'])

X = df.iloc[:,:11] #feature variable
y = df['high_quality'] #outcome variable

X = StandardScaler().fit_transform(X) #standard scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#model prediction
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print('lr_accuracy score={}'.format(lr_acc_score*100))
print(classification_report(y_test, lr_predict))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_predict = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print('dt_accuracy score={}'.format(dt_acc_score*100))
print(classification_report(y_test, dt_predict))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)
rfc_conf_matrix = confusion_matrix(y_test, rfc_predict)
rfc_acc_score = accuracy_score(y_test, rfc_predict)
print(rfc_conf_matrix)
print('rf_accuracy score={}'.format(rfc_acc_score*100))
print(classification_report(y_test, rfc_predict))

#Support Vector Machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_predict = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predict)
svc_acc_score = accuracy_score(y_test, svc_predict)
print(svc_conf_matrix)
print('svc_accuracy score={}'.format(svc_acc_score*100))
print(classification_report(y_test, svc_predict))

#Optimize parameters of SVC
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)

grid_svc.best_params_

svc2 = SVC(C = 1.2, gamma =  1, kernel= 'rbf')
svc2.fit(X_train, y_train)
svc2_predict = svc2.predict(X_test)
svc2_conf_matrix = confusion_matrix(y_test, svc2_predict)
svc2_acc_score = accuracy_score(y_test, svc2_predict)
print(svc2_conf_matrix)
print('accuracy score={}'.format(svc2_acc_score*100))
print(classification_report(y_test, svc2_predict))





