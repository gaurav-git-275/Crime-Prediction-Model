# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:07:53 2018

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
filename = 'New Kidnapping.csv'
names = ['Area','Month','Week','Time','Kidnapping_Crime']
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
pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png('tree_kidnapping.png')

#visualize way3
def show_tree(Mytree,path):
    f= io.StringIO()
    tree.export_graphviz(Mytree,out_file=f)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img=misc.imread(path)
    plt.rcParams["figure.figsize"]=(20,20)
    plt.imshow(img)
    
#show_tree(Mytree,'dec_tree01.png')
    
