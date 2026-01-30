# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and prepare the multivariate input data and target values.
2. Initialize the SGD Regressor with appropriate learning rate and iterations.
3. Train the model using the given dataset and predict the output values.
4. Compare actual and predicted values using graphical visualization.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ADITHYA NM
RegisterNumber: 212225040011
*/
```
```
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

# Sample data (2 features)
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])

# Create model
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')

# Train model
model.fit(X, y)

# Check learned weights
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# Predict
y_pred = model.predict(X)

# Plot Actual vs Predicted
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.show()
```

## Output:
<img width="1425" height="648" alt="ML-EX-4" src="https://github.com/user-attachments/assets/2990d9f7-05a3-475e-993d-dde6e61a8747" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
