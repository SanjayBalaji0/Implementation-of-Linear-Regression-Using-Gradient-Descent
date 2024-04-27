# Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.Sanjay Balaji
RegisterNumber:  212223240149
*/
```
```
 import numpy as np
 import pandas as pd
 from sklearn.preprocessing import StandardScaler
 def linear_regression(x1,y,learning_rate=0.1,num_iters=1000):
 x=np.c_[np.ones(len(x1)),x1]
 theta=np.zeros(x.shape[1]).reshape(-1,1)
 for _ in range(num_iters):
 prediction=(x).dot(theta).reshape(-1,1)
 errors=(prediction-y).reshape(-1,1)
 theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
 return theta   
```
```
data=pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv")
 data.head()
```
```
 x=(data.iloc[1:,:-2].values)
 x1=x.astype(float)
 scaler=StandardScaler()
 y=(data.iloc[1:,-1].values).reshape(-1,1)
 x1_Scaled=scaler.fit_transform(x1)
 y1_Scaled=scaler.fit_transform(y)
 print(x)
 print(x1_Scaled)
```
```
theta=linear_regression(x1_Scaled,y1_Scaled)
 new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
 new_Scaled=scaler.fit_transform(new_data)
 prediction=np.dot(np.append(1,new_Scaled),theta)
 prediction=prediction.reshape(-1,1)
 pre=scaler.inverse_transform(prediction)
 print(prediction)
 print(f"Predicted value: {pre}")
```

## Output:
### head()
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/25dfd0c6-3f18-4827-89b4-ae6ced65f507)
### x
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/118527ee-e9cf-49e6-9abc-e8d33ae6b75e)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/b13e1091-ba43-4af7-8a8b-88768cd7418e)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/c3f587c3-6002-4097-a36f-54022269e14c)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/668c8ef7-d999-448d-bf78-4a7eb8d9a542)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/d42a73e0-9947-434f-9bb0-44bae273cfe5)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/a0b32e44-3dc7-453d-8a42-3fc0efff0678)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/d4aad1bb-d769-4f24-8654-7e3acc3b6df4)
### Predicted value
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/8eaac0c0-719c-496f-8c90-d5619cd8d2f3)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
