# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:21:21 2018

@author: rafe
"""

import numpy as np
import pandas as pd
import seaborn as sb
import sklearn
import scipy

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

from sklearn import tree
import collections
import graphviz 
import pydot
import pydotplus
import io
from scipy import misc

#from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Load CSV using Pandas
filename = 'New Assult.csv'
names = ['Area','Day','Month','Time','Assult_Crime']
data = pd.read_csv(filename, names=names)
#print(data.head())
#print(data.info())
#print(data.shape)
#print(data.iloc[0].values)
X=data.iloc[:,0:4].values
y=data.iloc[:,-1].values
#counter=collections.Counter(y)
#print(counter)
#print(X)
#print(len(X))
#print(y)
#print(len(y))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)

#checking for independence between features
#sb.regplot(x='Month',y='Week',data=data, scatter=True)
mn=data['Month']
wk=data['Day']
spearmanr_coefficient,p_value=spearmanr(mn,wk)
print("spearmanr Rank correlation coeffient: %0.3f" % (spearmanr_coefficient))


logReg = LogisticRegression()
print(logReg)
logReg.fit(X_train,y_train)
y_pred=logReg.predict(X_test)
print("Logictic Regression  Prediction Accuracy (in %) :{:.3f}",accuracy_score(y_test,y_pred)*100)
print("Logistic Regression Model Score (in %): {:.3f}",logReg.score(X_train,y_train)*100)

#cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logReg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logReg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Assult_log_ROC')
plt.show()
