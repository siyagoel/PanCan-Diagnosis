#Phase 4: User Interface program

#Import all the libraries used in the User Interface (Tkinter, plotting, smoting, etc.) and libraries used for algorithms in the Ensemble Algorithm (KNN, Naive Bayes, Neural Network and Logistic Regression)
%reset
import re
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox  
from tkinter.filedialog import askopenfilename
import time
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import sklearn
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from warnings import simplefilter
import warnings
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
global a
regex = '^\w([\.-]?\w+)*@\w([\.-]?\w+)*(\.\w{2,3})+$'
def filedreq1():
    print ("Hello")

#Function filedreq will check if any field from the patient's personal information is left empty by user. If a field is empty, then it will give error.
def filedreq():
    if Fname.get() == "":
        print("First Name Field is Empty!!")
        user = "First Name Field is Empty!!"
        Label(win,text=user,fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=460)
    elif Lname.get() == "":
        print("Last Name Field is Empty!!")
        user = "Last Name Field is Empty!!"
        Label(win,text=user,fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=460)
    elif Email.get() == "":
        print("Email Field is Empty!!")
        user = "Email Field is Empty!!"
        Label(win,text=user,fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=460)
    elif Phone.get() == "":
        print("Phone Field is Empty!!")
        user = "Phone Field is Empty!!"
        Label(win,text=user,fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=460)
    elif entry_Address.compare("end-1c", "==", "1.0"):
        print("Address Field is Empty!!")
        user = "Address Field is Empty!!"
        Label(win,text=user,fg="blue",bg="yellow",font = ("Calibri 10 bold")).place(x=12,y=460)
    else:
        checkemail()

#Function checkemail will check if patient's email information is valid
def checkemail():
    email = Email.get()
    if(re.search(regex,email)):
        user = "Valid Email!!"
        test()
    else:
        print("Invalid Email!!")
        user = "Invalid Email-Type!! Type Correct Mail: Format: xyz@abc.com"
        MsgBox = tk.messagebox.showinfo ('information','Invalid Email-Type!! Type Correct Mail: Format: xyz@abc.com')

#Function browse allows user to upload a file with a .py extension from the local file system
def browse():
    filename = filedialog.askopenfilename(filetypes = (("All Files","*.*"),("File","*.py")))
    path.config(text = filename)
    
    a = filename
    global file 
    file = a

#Function import_csv_data allows user to upload a file with a .csv extension from the local file system
def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)

#Function Stacking implements a stacking algorithm technique. The stacking function takes several inputs, including a model, training and test data, the target variable y, the number of folds for cross-validation, and the type of model being used. The function returns the predictions made by the stacked model on the test set and the out-of-fold predictions made during cross-validation on the training set.
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


#Function algor is used to predict if patient has PC/No PC. It calls the stacking function for all the three algorithms used at Level 0 (K-Nearest Neighbors (KNN), Naive Bayes and Neural Network). Prediction is made at Level 1 using the Logistic Regression algorithm which takes into account the response from each of the three individual algorithms at Level 0.
def algor(x_train,y_train,x_test, y_test, n_neighbors,var_smoothing, Cs, iterations):
 
    #Finds best value for the hyperparameter n_neighbors using a KNN algorithm
    param_grid_knn=dict(n_neighbors=n_neighbors)
    knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn)
    knn.fit(x_train, y_train)
    n_neighborsmax=knn.best_estimator_.n_neighbors
    
    #Finds best value for the hyperparameter var_smoothing using a Naive Bayes algorithm
    param_grid_nb=dict(var_smoothing=var_smoothing)
    NaiveBayes=GridSearchCV(GaussianNB(), param_grid_nb)
    NaiveBayes.fit(x_train, y_train)
    var_smoothingmax=NaiveBayes.best_estimator_.var_smoothing
    
    #Performs stacking on the KNN algorithm using the best value for the hyperparameter n_neighbors
    param_grid_knn_final=dict(n_neighbors=[n_neighborsmax])
    model2 = GridSearchCV(KNeighborsClassifier(), param_grid_knn_final)
    test_pred2 ,train_pred2=Stacking(model=model2,train=x_train,y=y_train, test=x_test, n_fold=10, alg=1)
    train_pred2=pd.DataFrame(train_pred2)
    test_pred2=pd.DataFrame(test_pred2)
    
    #Performs stacking on the Naive Bayes algorithm using the best value for the hyperparameter var_smoothing
    param_grid_nb_final=dict(var_smoothing=[var_smoothingmax])
    model1=GridSearchCV(GaussianNB(), param_grid_nb_final)
    test_pred1 ,train_pred1=Stacking(model=model1,train=x_train,y=y_train, test=x_test, n_fold=10, alg=2)
    train_pred1=pd.DataFrame(train_pred1)
    test_pred1=pd.DataFrame(test_pred1)
    
    #Performs stacking on a Neural Network using three dense layers
    model3=Sequential()
    model3.add(Dense(25, activation= 'relu'))
    model3.add(Dense(15, activation= 'relu'))
    model3.add(Dense(1, activation= 'sigmoid'))
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_pred3 ,train_pred3=Stacking(model=model3,train=x_train,y=y_train, test=x_test, n_fold=10, alg=3)
    train_pred3=pd.DataFrame(train_pred3)
    test_pred3=pd.DataFrame(test_pred3)
    
    #Predictions are concatenated into dataframes df and df_test
    df = pd.concat([train_pred1, train_pred2, train_pred3], axis=1)
    df_test = pd.concat([test_pred1, test_pred2, test_pred3], axis=1)
    
    #Logistic regression algorithm is trained using hyperparameter tuning (GridSearch) and the final prediction is obtained
    param_grid_lr=dict(C=Cs,  max_iter=iterations)
    LogReg=GridSearchCV(LogisticRegression(penalty='l2'), param_grid_lr)
    LogReg.fit(df,y_train)
    y_predict = LogReg.predict(df_test)
    size=np.shape(y_predict)
    print (np.shape(y_predict))
    print(size)
    print(y_predict)
    print(size[0]-1)
    y_predDiab=y_predict[size[0]-1]
    print(y_predDiab)
    return y_predDiab

#Function test displays the PC results with stage along with the 5 most differentially expressed miRNAs
def test():
    #Data is read from a CSV file specified by the file path and is printed onto console for verification
    print("Testing...")
    print (Fname.get(), Lname.get(), Email.get(), Phone.get(),entry_Address.get(1.0,END))
    print (v.get())
    patient_Data=pd.read_csv(v.get())
    X_patient= patient_Data.iloc[:,1:191].values
    y_patient=patient_Data.iloc[:,191].values
    print(X_patient, y_patient)
    
    #Reads data and performs data splitting (80% training and 20% testing)
    Data_try=pd.read_csv("C:/files/micro_rna-200.csv")
    X = Data_try.iloc[1:,1:191].values
    yval = Data_try.iloc[1:,191].values
    x_train, x_test_temp, y_train, y_test_temp = train_test_split(X, yval, test_size=0.2)
   
   #Concatenates the testing dataset
    x_test=np.concatenate((x_test_temp,X_patient))
    y_test=np.concatenate((y_test_temp,y_patient))
    
    #Defines variables used for hyperparameter tuning and applies algor function to the training and testing data as well as the variables
    Cs=np.logspace(-2,1,10)
    n_neighbors=np.array([2,3,4,5,10,15, 20,25, 30, 32, 34])
    var_smoothing=np.logspace(-7,4,5)
    iterations=np.array([1000])
    y_predDiab=algor(x_train, y_train, x_test, y_test, n_neighbors, var_smoothing, Cs, iterations)
    
    #If patient has PC then data is read and X and y values are defined
    if y_predDiab == '1':
        Data_try_st=pd.read_csv("C:/files/stages-200.csv")
        X_st = Data_try_st.iloc[1:,1:191].values
        yval_st = Data_try_st.iloc[1:,191].values
        
        #Summarizes the new class distribution
        counter = Counter(yval_st)
        print(counter)
        
        #Transforms the dataset
        oversample = SMOTE(sampling_strategy=0.6)
        X_st, yval_st = oversample.fit_resample(X_st, yval_st)

        #Summarizes the new class distribution
        counter = Counter(yval)
        print(counter)

        #Splits data and concatenates test sets
        x_train_st, x_test_temp_st, y_train_st, y_test_temp_st = train_test_split(X_st, yval_st, test_size=0.2)
        x_test_st=np.concatenate((x_test_temp_st,X_patient))
        y_test_st=np.concatenate((y_test_temp_st,y_patient))
 
        #Data is read
        Data_try_st_avg=pd.read_csv("C:/files/stages-200-avg.csv")
        
        #variables are defined
        X_patient1=X_patient.astype(float)
        X_st_early_avg = Data_try_st_avg.iloc[101:102,1:191].values
        X_st_late_avg = Data_try_st_avg.iloc[102:103,1:191].values
        X_st_total_avg = Data_try_st_avg.iloc[103:104,1:191].values
        rnano=np.array([3, 14, 27, 30, 32, 45, 67, 70, 74, 77, 85, 94, 98, 112, 113, 119, 156, 157, 177, 187])
        error=np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
        mrna_names=np.array(['hsa-miR-30a-3p','hsa-miR-143-3p','hsa-miR-146b-5p','hsa-miR-520g-3p','hsa-miR-514a-3p','hsa-miR-645','hsa-miR-574-5p','hsa-miR-450b-3p','hsa-miR-934','hsa-miR-1283','hsa-miR-3133','hsa-miR-4310','hsa-miR-4272','hsa-miR-4455','hsa-miR-4460','hsa-miR-4508','hsa-miR-1271-3p','hsa-miR-1185-1-3p','hsa-miR-6823-3p','hsa-miR-8054'])
        patspecific_mrna=np.array(['xxxxxxxxxxxxxxxxx','xxxxxxxxxxxxxxxxx','xxxxxxxxxxxxxxxxx','xxxxxxxxxxxxxxxxx','xxxxxxxxxxxxxxxxx'])
        patspecific_percent=np.array(['xxxxx','xxxxx','xxxxx','xxxxx','xxxxx'])
        
        #Applies algor function to the training and testing data to predict stage of PC
        y_predDiab_stage=algor(x_train_st,y_train_st,x_test_st, y_test_st, n_neighbors,var_smoothing, Cs, iterations)
               
    #If patient has PC
    if y_predDiab == '1':
    
        #If patient is predicted to have stage 0 PC
        if y_predDiab_stage == '0':
            
            #Error 1 and error 2 are calculated
            countx=0
            for x in rnano:
                error1=np.absolute((X_st_early_avg[0,x]-X_patient1[0,x])/X_st_early_avg[0,x])
                error2=((X_patient1[0,x]-X_st_early_avg[0,x])/X_st_early_avg[0,x])
                error[0,countx]=countx
                error[1,countx]=error1*100.0
                error[2,countx]=error2*100.0
                countx=countx+1

            #Additional processing is performed on the error array to sort the data based on the second column, extract specific information, and store the results in the patspecific_mrna and patspecific_percent arrays.
            error_tr=np.transpose(error)
            sorting=error_tr[error_tr[:,1].argsort()]
            countx=0
            for x in range(0,5):
                mrna_no=sorting[countx,0].astype(int)
                patspecific_mrna[countx]=mrna_names[mrna_no]
                patspecific_percent[countx]=sorting[countx,2].astype(str)
                countx=countx+1
            
            #Error and sorting arrays are printed as well as the error rates and names for the top 5 most differentially expressed miRNAs.
            print(error)
            print(sorting)
            print(patspecific_mrna)
            print(patspecific_percent)
            
            #Creates a warning message box that displays information about the patient's microRNAs and associated percentage errors.
            msga="Five main MicroRNA involved for this patient are \n"+patspecific_mrna[0]+"   "+patspecific_percent[0]+"%\n"+patspecific_mrna[1]+"   "+patspecific_percent[1]+"%\n"+patspecific_mrna[2]+"   "+patspecific_percent[2]+"%\n"+patspecific_mrna[3]+"   "+patspecific_percent[3]+"%\n"+patspecific_mrna[4]+"   "+patspecific_percent[4]+"%"
            msgb='Pancreatic Cancer=YES; Stage=Early\nPlease take measures!!\n'
            msgtot=msgb+"\n"+msga
            MsgBox = tk.messagebox.showwarning ('warning', msgtot,icon = 'warning')
            person = Fname.get()
            user = person + ' is Diagnosed By Early Stage Pancreatic Cancer!!!\n'+msga
        
        #If patient is not predicted to have stage 0 PC
        else:
        
            #Error 1 and error 2 are calculated
            countx=0
            for x in rnano:
                error1=np.absolute((X_st_late_avg[0,x]-X_patient1[0,x])/X_st_late_avg[0,x])
                error2=((X_patient1[0,x]-X_st_late_avg[0,x])/X_st_late_avg[0,x])
                error[0,countx]=countx
                error[1,countx]=error1*100.0
                error[2,countx]=error2*100.0
                countx=countx+1
            
            #Additional processing is performed on the error array to sort the data based on the second column, extract specific information, and store the results in the patspecific_mrna and patspecific_percent arrays.
            error_tr=np.transpose(error)
            sorting=error_tr[error_tr[:,1].argsort()]
            countx=0
            for x in range(0,5):
                mrna_no=sorting[countx,0].astype(int)
                patspecific_mrna[countx]=mrna_names[mrna_no]
                patspecific_percent[countx]=sorting[countx,2].astype(str)
                countx=countx+1
                
            #Error and sorting arrays are printed as well as the error rates and names for the top 5 most differentially expressed miRNAs.
            print(error2)
            print(sorting)
            print(patspecific_mrna)
            print(patspecific_percent)
            
            #Creates a warning message box that displays information about the patient's microRNAs and associated percentage errors.
            msga="Five main MicroRNA involved for this patient are \n"+patspecific_mrna[0]+"   "+patspecific_percent[0]+"%\n"+patspecific_mrna[1]+"   "+patspecific_percent[1]+"%\n"+patspecific_mrna[2]+"   "+patspecific_percent[2]+"%\n"+patspecific_mrna[3]+"   "+patspecific_percent[3]+"%\n"+patspecific_mrna[4]+"   "+patspecific_percent[4]+"%"            
            msgb='Pancreatic Cancer=YES; Stage=Late\nPlease take measures!!\n'
            msgtot=msgb+"\n"+msga
            MsgBox = tk.messagebox.showwarning ('warning', msgtot ,icon = 'warning')
            person = Fname.get()
            user = person + ' is Diagnosed By Late Stage Pancreatic Cancer!!!\n'+msga
        a = user

    #If patient does not have PC, creates a message box that says patient is not diagnosed with PC.
    else:
        person = Fname.get()
        user = person + ' is NOT  Diagnosed By Pancreatic Cancer!!!'
        a = user
        MsgBox = tk.messagebox.showinfo ('information','Pancreatic Cancer=NO \n No need to worry \n Have a Healthy Life Style!!')
    save(a)
    
#Function save saves the text file with patient information in Home Directory for future use
def save(a):
    First = Fname.get()
    Last = Lname.get()
    email = Email.get()
    address = entry_Address.get(1.0,END)
    phone = Phone.get()
    age=Age.get()
    gender = Gender.get()
    save_name = "C:/Users/j68/"+Last+"_"+First+"_"+age+".txt"
    file = open(save_name,"a")
    file.write("\n\nFirst Name: "+First+"\n")
    file.write("Last Name: "+Last+"\n")
    file.write("Phone: "+phone+"\n")
    file.write("Email: "+email+"\n")
    file.write("Address: "+address)
    file.write("Age: "+age+"\n")
    file.write("Gender: "+gender+"\n")
    file.write("Report: "+ a +"\n")
    file.close()
    MsgBox = tk.messagebox.showinfo ('information','Patient Health Detection have successfully done \n Report is saved in the text file by Patient name')

#Clears the values of various GUI elements to their initial or empty states
def reset():
    Fname.set("")
    Lname.set("")
    Email.set("")
    Phone.set("")
    Age.set("")
    Gender.set("")
    entry_Address.delete(1.0,END)

#Setup of Graphical User Interface (GUI) using Tkinter in Python
win =  Tk()

#Sets up background of GUI
win.geometry("800x600")
win.configure(background="light blue")
win.title("Pancreatic Cancer Diagnosis Tool By Siya Goel")
title = Label(win,text="PANCAN DIAGNOSIS",bg="purple",width="300",height="2",fg="White",font = ("Calibri 28 bold")).pack()
my_img = ImageTk.PhotoImage(Image.open("C:/Users/j68/Picture1.tif"))
my_label = Label(image=my_img)
my_label.place(x=700,y=5)
background_image=ImageTk.PhotoImage(Image.open("C:/Users/j68/PC1.tif"))
background_label = Label(image=background_image)
background_label.place(x=315, y=110)

#Creates labels for boxes where people insert information
Fname = Label(win, text="First name: ",bg="light blue",fg="Black", font = ("Verdana 12")).place(x=12,y=120)
Lname = Label(win, text="Last name: ",bg="light blue",fg="Black",font = ("Verdana 12")).place(x=12,y=160)
email = Label(win, text="Email ID: ",bg="light blue",fg="Black",font = ("Verdana 12")).place(x=12,y=200)
Phone = Label(win, text="Phone: ",bg="light blue",fg="Black",font = ("Verdana 12")).place(x=12,y=240)
Address = Label(win, text="Address: ",bg="light blue",fg="Black",font = ("Verdana 12")).place(x=12,y=280)
Age = Label(win, text="Age: ",bg="light blue",fg="Black",font = ("Verdana 12")).place(x=12,y=340)
Gender = Label(win, text="Gender(M/F): ",bg="light blue",fg="Black",font = ("Verdana 12")).place(x=12,y=380)

#Stores and manages strings within each field of information by creating variables
Fname = StringVar()
Lname = StringVar()
Email = StringVar()
Phone = StringVar()
Address  = StringVar()
Age= StringVar()
Gender = StringVar()
Courses  = StringVar()

#Each variable is matched with an input field to manage the data
entry_Fname = Entry(win,textvariable = Fname,width=30)
entry_Fname.place(x=120,y=120)
entry_Lname = Entry(win,textvariable = Lname,width=30)
entry_Lname.place(x=120,y=160)
entry_email = Entry(win,textvariable = Email,width=30)
entry_email.place(x=120,y=200)
entry_Phone = Entry(win,textvariable = Phone,width=30)
entry_Phone.place(x=120,y=240)
entry_Address = Text(win,height=2,width=23)
entry_Address.place(x=120,y=280)
entry_Age = Entry(win,textvariable = Age,width=30)
entry_Age.place(x=120,y=340)
entry_Gender = Entry(win,textvariable = Gender,width=30)
entry_Gender.place(x=120,y=380)

#Completes the GUI by implementing upload, reset, and submit buttons
v = tk.StringVar()
path = Label(win,bg="light blue",font = ("Verdana 8"))
path.place(x=140,y=400)
upload = Button(win, text="Browse Data Set", width="16",height="1",activebackground="blue", bg="Pink",font = ("Calibri 12 "),command = import_csv_data).place(x=20, y=420)
reset = Button(win, text="Reset", width="12",height="1",activebackground="red", bg="Pink",font = ("Calibri 12 "),command = reset).place(x=20, y=470)
submit = Button(win, text="Test", width="12",height="1",activebackground="violet", bg="Pink",command = filedreq,font = ("Calibri 12 ")).place(x=240, y=470)
win.mainloop()
