# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Python Libraries and Load The iris dataset

2.Set the target value

3.Split Dataset into Training and Testing Sets

4.Train the Model Using Stochastic Gradient Descent (SGD) 

5.Make Predictions,Evaluate Accuracy And Confusion Matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VINOTH M P
RegisterNumber:  212223240182
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd=SGDClassifier(max_iter=1000,tol=1e-3)
sgd.fit(x_train,y_train)

y_pred=sgd.predict(x_test)
print("Prediction:\n",y_pred)

accu=accuracy_score(y_test,y_pred)
print("Accuracy:\n",accu)

confu=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confu)
```

## Output:
![image](https://github.com/user-attachments/assets/6e04323e-c036-4cd5-b050-a010c3ea43e9)

![image](https://github.com/user-attachments/assets/53c565af-d46c-407e-8ddc-0fbe864de797)

![image](https://github.com/user-attachments/assets/f30fee91-20bd-4110-a5c2-0641ac4dcc6d)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
