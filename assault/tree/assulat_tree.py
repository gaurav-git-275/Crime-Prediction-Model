# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:01:48 2018

@author: Rafe Nakhuda
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns
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
names = ['Area','Month','Week','Time','Assult_Crime']
data = pd.read_csv(filename, names=names)
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

Mytree=DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=30)
Mytree.fit(X_train,y_train)
print(Mytree)
#y_pred=tree.predict(X_test)
#print("Decision Tree model accuracy(in %) {:.3f}:",accuracy_score(y_test,y_pred)*100)
print("Decision Tree Accuracy in %: {:.3f}".format(Mytree.score(X_train,y_train)*100))

#visualize way1
#dotfile =io.StringIO()
#dot_data = tree.export_graphviz(Mytree, out_file=dotfile) 
#graph = graphviz.Source(dot_data) 
#graph.render("crime") 

#visualize way2
dotfile =io.StringIO()
tree.export_graphviz(Mytree, out_file=dotfile)
pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png('tree_assult.png')

#visualize way3
def show_tree(Mytree,path):
    f= io.StringIO()
    tree.export_graphviz(Mytree,out_file=f)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img=misc.imread(path)
    plt.rcParams["figure.figsize"]=(20,20)
    plt.imshow(img)
    
#show_tree(Mytree,'dec_tree01.png')
    
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, Mytree.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, Mytree.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()