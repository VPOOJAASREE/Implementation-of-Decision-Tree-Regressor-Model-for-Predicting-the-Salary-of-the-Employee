# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: V.POOJAA SREE
RegisterNumber: 212223040147 
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:


![o1](https://github.com/user-attachments/assets/29edd43a-c024-4d41-a78f-4b75fc4f6dab)


![o2](https://github.com/user-attachments/assets/adf74861-143c-4ad8-8513-30c84f7f6e56)


![o3](https://github.com/user-attachments/assets/7fe3c9e0-7711-4b69-ac56-8732f6829c75)


![o4](https://github.com/user-attachments/assets/4a61c08b-d57c-40c4-8173-35e39d9b1e58)


![o5](https://github.com/user-attachments/assets/ee897dfe-918e-4fa8-93db-ad8deb589bdd)


![o6](https://github.com/user-attachments/assets/5fe0dc17-3328-4470-9cbd-a7b0c89546d2)


![o7](https://github.com/user-attachments/assets/60696a16-becb-4edd-a738-18005b5d027e)


![o8](https://github.com/user-attachments/assets/c242ce72-f033-49d8-aa24-73d22dd12ebd)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
