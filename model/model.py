import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import pickle
import warnings 
warnings.simplefilter("ignore")

    
#Algorithms.
#Support Vector Machine
def svc(dataset):
    SVC_obj=SVC(kernel='linear',gamma='auto',C=2)          
    SVC_obj.fit(x_train,y_train)
    #Predict the outcome
    y_predict_SVC=SVC_obj.predict(x_test)
    print(y_predict_SVC)
    print(classification_report(y_test,y_predict_SVC))
    print("Accuracy percentage SVC:"+"{:.2f}".format(accuracy_score(y_test,y_predict_SVC)*100))
    a=accuracy_score(y_test,y_predict_SVC)
    print(a)
    #Confusion matrix
    cm = confusion_matrix(y_test, y_predict_SVC)
    print(cm)
    return a

#DecisionTreeClassifier  
def DTC(dataset):
    DTC_obj=DecisionTreeClassifier()     
    DTC_obj.fit(x_train,y_train)
    #Predict the outcome
    y_predict_DTC=DTC_obj.predict(x_test)
    print(classification_report(y_test,y_predict_DTC))
    print("Accuracy percentage DTC:"+"{:.2f}".format(accuracy_score(y_test,y_predict_DTC)*100))
    b=accuracy_score(y_test,y_predict_DTC)
    #Confusion matrix
    cm = confusion_matrix(y_test, y_predict_DTC)
    print(cm)
    return b

#Logistic Regression
def LR(dataset):
    LR_obj=LogisticRegression()      
    LR_obj.fit(x_train,y_train)
    #Predict the outcome
    y_predict_LR=LR_obj.predict(x_test)
    print(classification_report(y_test,y_predict_LR))
    print("Accuracy percentage LR:"+"{:.2f}".format(accuracy_score(y_test,y_predict_LR)*100))
    d=accuracy_score(y_test,y_predict_LR)
    #Confusion matrix
    cm = confusion_matrix(y_test, y_predict_LR)
    print(cm)
    return d
#KNeighborsClassifier
def KNC(dataset):
    KNC_obj=KNeighborsClassifier()      
    KNC_obj.fit(x_train,y_train)
    #Predict the outcome
    y_predict_KNC=KNC_obj.predict(x_test)
    print(classification_report(y_test,y_predict_KNC))
    print("Accuracy percentage KNC:"+"{:.2f}".format(accuracy_score(y_test,y_predict_KNC)*100))
    e=accuracy_score(y_test,y_predict_KNC)
    #Confusion matrix
    cm = confusion_matrix(y_test, y_predict_KNC)
    print(cm)
    return e


dataset=pd.read_csv('./documents/data.csv')

#Data Cleaning
dataset.isnull().sum()#Get the count of no.of null values 

dataset.describe()#Summary after cleaning data

dataset['Crop'].fillna(str(dataset['Crop'].mode().values[0]),inplace=True)

#Categorical to Numerical
dataset['Crop']=dataset['Crop'].map({"Wheat":0,"Rice":1,"Maize":2,"GreenGram":3,"Pea":4,
                                     "Pigeon Pea":5,"Sunflower":6,"Onion":7,"Millet":8,
                                     "Potato":9,"Sugarcane":10,"Cotton":11,"SoyaBean":12,"Tomato":13})
#dataset['Crop'] = dataset.Crop.astype(int)
dataset['pH'] = dataset.pH.astype(int)
print(dataset)
print(dataset.Crop.unique())


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2,random_state=10)

y=dataset.Value
x=dataset.drop('Value',axis=1)

dtc=DecisionTreeClassifier()

#Train the model
dtc.fit(x_train,y_train)

# Saving model to disk
pickle.dump(dtc, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))




    
