# Multiple Linear Regression from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 11SEP20

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[: , :-1].values  # matrix of features -> predictors
y = dataset.iloc[:, -1].values    # independent variable

print(X)

# Encode categories
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

print(X)

# No need to apply feature scaling in multiple linear regression as the coefficients on each 
# independent variable will compensate to put everything on the same scale

# Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train the Multiple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results