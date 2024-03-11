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
Developed by: 
RegisterNumber:  
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
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/896598f3-6f61-4004-b4bd-5eebadd77124)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/238f0c53-4eab-46a9-8018-aeb2f17ff47b)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/772c1974-1a09-4eb8-ab74-08994d08db09)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/e8611d63-8fa5-476b-b098-27768c7257b7)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/98da229e-54ea-4773-866c-934082d8b6ad)
![image](https://github.com/SanjayBalaji0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145533553/006dfdcb-88cb-4747-bc72-3e778bd7d8ef)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
