# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:19:46 2018

@author: siddi
"""

#naive Bays Classifier on area year month day time over property_crime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import urllib
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Load CSV using Pandas
filename = 'New Assult.csv'
names = ['Area_ID','Day', 'Month','Year','Time','Assault']
data = pd.read_csv(filename, names=names)
#print(data.shape)
#print(data.iloc[0].values)
X=data.iloc[:,0:5].values
y=data.iloc[:,-1].values
#print(X)
#print(len(X))
#print(y)
#print(len(y))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)

x_testUser=[[2,23,4,2013,3]]
y_testUser=["Yes"]
#print(x_testUser)
#print(y_testUser)
#ynew_test=["Yes","No"]
#print(ynew_test)

MultiNB=MultinomialNB()
MultiNB.fit(X_train,y_train)
print(MultiNB)
y_pred=MultiNB.predict(x_testUser)
#print(y_pred)
y_predProb=MultiNB.predict_proba(x_testUser)
print("Probabilty of Happening Assault for your input (Yes,No) :")
print(y_predProb)
print("Accuracy(in %):",accuracy_score(y_testUser,y_pred)*100)
