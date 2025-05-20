# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
/*
Program to implement the SVM For Spam Mail Detection..

Developed by: DINESH PRABHU S

RegisterNumber: 212224040077
*/
```
import chardet
file = "/content/spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv("/content/spam.csv", encoding='windows-1252')
print(data.head())

print(data.info())

print(data.isnull().sum())

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:
![image](https://github.com/user-attachments/assets/c9988113-c93a-4c9a-8f0b-bf668f85678d)

![image](https://github.com/user-attachments/assets/939ece05-51df-4a76-a14f-c3ed46cceb46)

![image](https://github.com/user-attachments/assets/eb9cda3a-758d-47a9-b7d7-1cf572168aa6)

![image](https://github.com/user-attachments/assets/69ee0298-d7e2-4f45-a309-05ed535e116d)

![image](https://github.com/user-attachments/assets/87e1ec67-d87c-47f4-b7b5-d80f9f12e674)

![image](https://github.com/user-attachments/assets/5f194d55-b616-48f6-8111-9e5a6fe5e753)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
