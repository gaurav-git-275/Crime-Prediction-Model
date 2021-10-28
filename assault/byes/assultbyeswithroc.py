# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:12:14 2018

@author: Rafe Nakhuda
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
from matplotlib import pyplot as plt

#from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Load CSV using Pandas
filename = 'New Assult.csv'
names = ['Area_ID','Day', 'Month','Year','Time','Propery_Crime']
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
print(y_test)

MultiNB=MultinomialNB()
MultiNB.fit(X_train,y_train)
print(MultiNB)
y_pred=MultiNB.predict(X_test)
print(y_pred)
print("Multinomial Naive Bayes model accuracy(in %):",accuracy_score(y_test,y_pred)*100)
results = confusion_matrix(y_test,y_pred)
#print("Confusion Matrix :")
print(results)
#print("Accuracy Score :",accuracy_score(y_test, y_pred))
print("Report :")
print(classification_report(y_test,y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, MultiNB.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, MultiNB.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Assult_ROC')
plt.show()



