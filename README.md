# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Standardize the data: This scales the features (x1) and target variable (y) by subtracting the mean and dividing by the standard deviation.
2.Train the linear regression model: This calculates the theta values (model parameters) using gradient descent. Here, the model updates the thetas by multiplying the learning rate with the product of the transpose of the feature matrix (x) and the errors (difference between predicted and actual values).
3.Scale the new data: The new data point (new_data) is scaled using the same scaler used for the training data.
4.Predict the value: The predicted value is calculated by multiplying the scaled new data with the trained theta values, and then inverting the scaling to get the original scale.
  
  


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: BALAJEE KS
RegisterNumber:  212222080009
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
  return theta
data=pd.read_csv("/content/50_Startups.csv",header=None)
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
Scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_Scaled=Scaler.fit_transform(x1)
y1_Scaled=Scaler.fit_transform(y)
theta=linear_regression(x1_Scaled,y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=Scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=Scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")
*/
```

## Output:
![linear regression using gradient descent](sam.png)
![Screenshot (446)](https://github.com/balajeeakm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/131589871/83513666-9634-456e-8c67-37b284a92589)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
