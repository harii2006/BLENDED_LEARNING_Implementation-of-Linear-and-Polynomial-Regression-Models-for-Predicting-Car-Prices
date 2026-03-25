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
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment (1).csv')
df.head()
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price'] 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scalar=StandardScaler()
x_train_scaled=scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
print('Name:SHRIHARI M')
print('Reg. No:212225230265')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}:{mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}:{np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-squared':>12}:{r2_score(y_test, y_pred):>10.2f}")
print(f"{'MAE':>12}:{mean_absolute_error(y_test, y_pred):>10.2f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals=y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
     "\n(Values close to 2 indicate no autocorrelation)")
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()

```

## Output:
<img width="205" height="246" alt="Screenshot 2026-03-25 141939" src="https://github.com/user-attachments/assets/4bd8c990-a425-4d82-a663-da53a5a398f7" />

<img width="793" height="426" alt="Screenshot 2026-03-25 141957" src="https://github.com/user-attachments/assets/a9f93dd8-517a-44bd-803e-00acfafeb9b9" />

<img width="798" height="485" alt="Screenshot 2026-03-25 142010" src="https://github.com/user-attachments/assets/13ca6792-d641-43f1-af36-f1b7ecb156d1" />

<img width="810" height="355" alt="Screenshot 2026-03-25 142023" src="https://github.com/user-attachments/assets/3ba37a70-1288-4e89-b8f4-8a0f9b7d07f2" />

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
