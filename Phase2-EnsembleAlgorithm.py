#Phase 2: Code for Pancreatic Cancer detection

#Import all the libraries to be used in Ensemble Algorithm (KNN, Naive Bayes, Neural Network and Logistic Regression) along with numpy and pandas
%reset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import sklearn
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from warnings import simplefilter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from warnings import simplefilter
import warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

#Function used to calculate training and testing prediction for Level 0 algorithms using StratifiedKFold
def Stacking(model,train,y,test,n_fold, alg):
    folds=StratifiedKFold(n_splits=n_fold,random_state=None, shuffle=False)
    test_pred=np.empty((0,1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y):
        x_train,x_val=train[train_indices],train[val_indices]
        y_train,y_val=y[train_indices],y[val_indices]
        if alg == 3:
            model.fit(x_train,y_train, epochs=10, verbose=0)
            train_pred=np.append(train_pred,model.predict_classes(x_val))
        else:
            model.fit(x_train,y_train)
            train_pred=np.append(train_pred,model.predict(x_val))
    if alg == 3:
        model.fit(train,y, epochs=10, verbose=0)
        test_pred=model.predict_classes(test)
    else:
        model.fit(train,y)
        test_pred=model.predict(test)   
    return test_pred,train_pred

#Reading in the data and defining important variables for training, testing, hypertuning, and evaluating the model (metrics).
Data_try=pd.read_csv("C:/files/micro_rna-200.csv")
X = Data_try.iloc[1:,1:191].values
yval = Data_try.iloc[1:,191].values
n_iterations = 40
n_iterations1 = 1
Cs=np.logspace(-2,1,10)
iterations=np.array([1000])
n_neighbors=np.array([2,3,4,5,10,15, 20,25, 30, 32, 34])
score_value1=np.zeros(n_iterations)
optimised_C=np.zeros(n_iterations)
scores=np.zeros(n_iterations)
F1=np.zeros(n_iterations)
F2=np.zeros(n_iterations)
Precision=np.zeros(n_iterations)
Recall=np.zeros(n_iterations)
specificity=np.zeros(n_iterations)
sensitivity=np.zeros(n_iterations)
TN=np.zeros(n_iterations)
TP=np.zeros(n_iterations)
FN=np.zeros(n_iterations)
FP=np.zeros(n_iterations)
rocs=np.zeros(n_iterations)
nneighbormax_value=np.zeros(n_iterations)
Cmax_value=np.zeros(n_iterations)
var_smoothingmax_value=np.zeros(n_iterations)
C_value=np.zeros(n_iterations1)
score_value_logreg=np.zeros(n_iterations1)
n_neighbors_value=np.zeros(n_iterations1)
score_value_knn=np.zeros(n_iterations1)
var_smoothing_value=np.zeros(n_iterations1)
score_value_naive=np.zeros(n_iterations1)
rocs=np.zeros(n_iterations)
matthew=np.zeros(n_iterations)
avgprecisionx=np.zeros(n_iterations)
jaccard=np.zeros(n_iterations)
dist=np.zeros(n_iterations)
pears=np.zeros(n_iterations)
spearm=np.zeros(n_iterations)
mutualscore=np.zeros(n_iterations)
var_smoothing=np.logspace(-7,4,5)

#Calculates the avaerage values of elements in vB0 based on duplicate elements in vA0.
def avg_group(vA0, vB0):
    vA, ind, counts = np.unique(vA0, return_index=True, return_counts=True)
    vB = vB0[ind]
    for dup in vA[counts>1]:
        vB[np.where(vA==dup)] = np.average(vB0[np.where(vA0==dup)])	
    return vA, vB

#Ensemble algorithm contains an outer loop for testing and inner loop for hypertuning the algorithm.
for i in range(n_iterations):
    print(i)
    for j in range(n_iterations1):
        
        #Split dataset into training (80% data) and testing (20% data)
        x_train, x_test, y_train, y_test = train_test_split(X, yval, test_size=0.2)

        # Hyperparameter Tuning for the models using GridSearchCV
        param_grid=dict(n_neighbors=n_neighbors)
        knn = GridSearchCV(KNeighborsClassifier(), param_grid)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        score_val_knn = accuracy_score(y_test, y_pred)
        optimised_n_neighbors=knn.best_estimator_.n_neighbors
        n_neighbors_value[j]=optimised_n_neighbors
        score_value_knn[j]=score_val_knn
        param_grid=dict(var_smoothing=var_smoothing)
        NaiveBayes=GridSearchCV(GaussianNB(), param_grid)
        NaiveBayes.fit(x_train, y_train)
        print(NaiveBayes.best_params_)
        y_pred = NaiveBayes.predict(x_test)
        score_val_naive = accuracy_score(y_test, y_pred)
        optimised_var_smoothing=NaiveBayes.best_estimator_.var_smoothing
        optimised_score=score_val_naive
        var_smoothing_value[j]=optimised_var_smoothing
        score_value_naive[j]=optimised_score
    
    #Split dataset into training (80% data) and testing (20% data)
    x_train, x_test, y_train, y_test = train_test_split(X, yval, test_size=0.2)

    #Stratified K fold algorithm is called for each of the Level 0 algorithms and training and testing prediction for each model is found

    #Model 1: KNN model to get a prediction at Level 0
    vA_n_neighbors, ind_n_neighbors, counts_n_neighbors = np.unique(n_neighbors_value, return_index=True, return_counts=True)
    (n_neighbors_final,score_final)=avg_group(n_neighbors_value, score_value_knn)
    n_neighbors_calc=score_final*counts_n_neighbors
    n_neighbors_index= counts_n_neighbors.argmax()
    n_neighborsmax1=n_neighbors_final[n_neighbors_index]
    n_neighborsmax=n_neighborsmax1.astype(int)
    nneighbormax_value[i]=n_neighborsmax
    param_grid_final=dict(n_neighbors=[n_neighborsmax])
    model2 = GridSearchCV(KNeighborsClassifier(), param_grid_final)
    test_pred2 ,train_pred2=Stacking(model=model2,train=x_train,y=y_train, test=x_test, n_fold=10, alg=1)
    train_pred2=pd.DataFrame(train_pred2)
    test_pred2=pd.DataFrame(test_pred2)

    #Model 2: Naive Bayes model to get a prediction at Level 0
    vA_var_smoothing, ind_var_smoothing, counts_var_smoothing = np.unique(var_smoothing_value, return_index=True, return_counts=True)
    (var_smoothing_final,score_final)=avg_group(var_smoothing_value, score_value_naive)
    var_smoothing_calc=score_final*counts_var_smoothing
    var_smoothing_index= var_smoothing_calc.argmax()
    var_smoothingmax=var_smoothing_final[var_smoothing_index]
    var_smoothingmax_value[i]=var_smoothingmax
    param_grid_final=dict(var_smoothing=[var_smoothingmax])
    model1=GridSearchCV(GaussianNB(), param_grid_final)
    test_pred1 ,train_pred1=Stacking(model=model1,train=x_train,y=y_train, test=x_test, n_fold=10, alg=2)
    train_pred1=pd.DataFrame(train_pred1)
    test_pred1=pd.DataFrame(test_pred1)
        
    #Model 3: Neural Network model to get a prediction at Level 0
    model3=Sequential()
    model3.add(Dense(25, activation= 'relu'))
    model3.add(Dense(15, activation= 'relu'))
    model3.add(Dense(1, activation= 'sigmoid'))
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_pred3 ,train_pred3=Stacking(model=model3,train=x_train,y=y_train, test=x_test, n_fold=10, alg=3)
    train_pred3=pd.DataFrame(train_pred3)
    test_pred3=pd.DataFrame(test_pred3)

    #The three model predictions are concatenated
    df = pd.concat([train_pred1, train_pred2, train_pred3], axis=1)
    df_test = pd.concat([test_pred1, test_pred2, test_pred3], axis=1)

    #Level 1 prediction using Logistic Regression model. Hyperparameter tuning is done and best estimator is used for prediction
    param_grid=dict(C=Cs,  max_iter=iterations)
    LogReg=GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
    LogReg.fit(df,y_train)
    scores_temp=LogReg.score(df_test, y_test)
    scores[i]=scores_temp
    optimised_C[i]=LogReg.best_estimator_.C
    y_predict = LogReg.predict(df_test)
    scorelist = accuracy_score(y_test, y_predict)
    score_value1[i]=scorelist

    #Several metrics like sensitivity, specificity, recall, precision, F1, and F2 scores are calculated
    cm=confusion_matrix(y_test, y_predict)
    TN[i]=cm[0,0]
    TP[i]=cm[1,1]
    FN[i]=cm[1,0]
    FP[i]=cm[0,1]
    sensitivity[i]=TP[i]/float(TP[i]+FN[i])
    specificity[i]=TN[i]/float(TN[i]+FP[i])
    Recall[i]=sensitivity[i]
    Precision[i]=TP[i]/float(TP[i]+FP[i])
    F1[i]=(1+1*1)*((Precision[i]*Recall[i])/((1*1*Precision[i])+Recall[i]))
    F2[i]=(1+2*2)*((Precision[i]*Recall[i])/((2*2*Precision[i])+Recall[i]))
    y_pred_prob_yes=LogReg.predict_proba(df_test)
    rocs[i]=sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])
    y_val1=y_test.astype(int)
    fpr, tpr, _ = metrics.roc_curve(y_val1,y_pred_prob_yes[:,1])
    auc = metrics.roc_auc_score(y_val1,y_pred_prob_yes[:,1])
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

#Average of metrics like sensitivity, specificity, recall, precision, F1, and F2 scores are calculated.
def Average(scorelist1): 
    return sum(scorelist1) / len(scorelist1)

print (np.mean(score_value1))
print (np.mean(scores))
print (np.mean(F1))
print (np.mean(F2))

