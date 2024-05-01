# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.

2. Import the dataset to operate on.

3. Split the dataset.

4. Predict the required output

## Program:

/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ARSHITHA MS
RegisterNumber:  212223240015
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


## Output:
### Output Result:
![image](https://github.com/23008344/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742655/79c65686-cbeb-4462-b595-02f2ca9c8784)

### Data.head(): 
![image](https://github.com/23008344/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742655/ab1d7145-7a53-4c89-87d4-5b7f487ca64f)

### Data.info():
![image](https://github.com/23008344/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742655/f71ff865-8d81-4652-9290-2d7c0414ab3b)

### Y_prediction value:
![image](https://github.com/23008344/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742655/46f653f2-6ba0-4dfa-90df-7f209e2af2cc)

### Accuracy value:
![image](https://github.com/23008344/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742655/64b5a271-40b4-4ac5-8113-b79ff5ff67c1)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
