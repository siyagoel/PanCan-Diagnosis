#Phase 1: Code for feature selection from 250 features to 100 features using different feature selection methods

#This code finds 100 features from PC and No PC Datasets as well as 100 features from Early/Late Stage PC
%reset
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
 
#Provides the csv file for the different feature selection methods (PC and No PC Datasets csv file or Early/Late Stage PC csv file to get 100 features)
NoDiab = pd.read_csv("C:/files/feature_selection.csv")
NoDiab.head()
X = NoDiab.iloc[:,1:2541].values
y = NoDiab.iloc[:,2541].values
X_feature=np.zeros((1328, 250))
aaa=NoDiab.iloc[:,2542].values
for count in range(250):
    X_feature[:,count]=X[:,aaa[count]]

#Obtains a list of models to evaluate
def get_models():
    models = dict()
    #Logistic Regression
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=100)
    model = LogisticRegression()
    models['lr'] = Pipeline(steps=[('s',rfe),('m',model)])
    #Perceptron
    rfe = RFE(estimator=Perceptron(), n_features_to_select=100)
    model = LogisticRegression()
    models['per'] = Pipeline(steps=[('s',rfe),('m',model)])
    #Cart
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=100)
    model = LogisticRegression()
    models['cart'] = Pipeline(steps=[('s',rfe),('m',model)])
    #Random Forest
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=100)
    model = LogisticRegression()
    models['rf'] = Pipeline(steps=[('s',rfe),('m',model)])
    #Gradient Boosting
    rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=100)
    model = LogisticRegression()
    models['gbm'] = Pipeline(steps=[('s',rfe),('m',model)])
    #Support Vector Machine
    rfe = RFE(estimator=SVC(), n_features_to_select=100)
    model = LogisticRegression()
    models['SVM'] = Pipeline(steps=[('s',rfe),('m',model)])
    return models
 
#Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
 
#Obtain the models to evaluate using get_models function
models = get_models()

#Evaluate the models and store results for each of the model
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X_feature, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

#Plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
