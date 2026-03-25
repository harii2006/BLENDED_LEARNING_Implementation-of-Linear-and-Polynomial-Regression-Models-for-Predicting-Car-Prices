# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset.
2.Select input features (enginesize, horsepower, citympg, highwaympg) and target (price).
3.Split the data into training and testing sets.
4.Create and train the Linear Regression model with scaling.
5.Create and train the Polynomial Regression model (degree = 2).
6.Predict prices, evaluate using MSE/MAE/R², and plot actual vs predicted values. 

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: SHRIHARI M
RegisterNumber:  212225230265
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#Load data 
df=pd.read_csv('encoded_car_data (1).csv')
print(df.head())

#Select features and targer
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#1.Linear Regression (with scaling)
lr=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(x_train,y_train)
y_pred_linear=lr.predict(x_test)

#2. Polynomial Regression (degree=2)
poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler())
,
    ('model',LinearRegression())
])
poly_model.fit(x_train,y_train)
y_pred_poly=poly_model.predict(x_test)

#Evaluate Models
print('Name:SHRIHARI M  ')
print('Red. No: 212225230265')
print('Linear Regression:')
#mse=mean_squared_error(y_test,y_pred_linear)
print('MSE=',mean_squared_error(y_test,y_pred_linear))
print('MAE=',mean_absolute_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)
print('R2 Score=',r2score)

#print (f"MSE {mean_squared_error(y_test,y_pred_linear):.2f}")
#print(f"R2: {r2_score(y_test,y_pred_linear):.2f"})

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test,y_pred_poly):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_poly):.2f}")
print(f"R2: {r2_score(y_test,y_pred_poly):.2f}")

#Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(y_test,y_pred_poly, label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

```

## Output:
<img width="812" height="571" alt="Screenshot 2026-03-25 144154" src="https://github.com/user-attachments/assets/20687bc3-7cb9-4db5-9fb6-b393272dc573" />


<img width="911" height="478" alt="Screenshot 2026-03-25 144201" src="https://github.com/user-attachments/assets/3cbee1d3-94a1-4bbe-bf16-5f6186a31ff4" />

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
