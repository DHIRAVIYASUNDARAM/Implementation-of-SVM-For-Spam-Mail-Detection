# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect and label email dataset as spam or non-spam.
2. Preprocess the text (remove stopwords, punctuation, lowercase, etc.).
3. Convert text to numerical features using TF-IDF or Bag of Words.
4. Split the dataset into training and testing sets.
5. Train an SVM classifier using the training data.
6. Evaluate the model on test data using accuracy, precision, and recall.


## Program and Output:

Program to implement the SVM For Spam Mail Detection.

Developed by: DHIRAVIYA S

RegisterNumber:  212223040041

```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
```
![image](https://github.com/user-attachments/assets/f5bf51f8-e365-448e-83b1-3054d29683c9)

```
import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
data.head()
```
![image](https://github.com/user-attachments/assets/82eced06-3789-4260-810a-9bf630992302)

```
data.info()

data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/1f735b99-d65b-4f71-9fc3-77580615607d)

```
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
```
![image](https://github.com/user-attachments/assets/4fbd6fbb-1e84-4463-881b-ea440983eaa4)


```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/2f8f8764-2fd2-47e7-8743-0350d902091f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
