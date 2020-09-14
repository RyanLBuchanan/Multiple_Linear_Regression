# Multiple Linear Regression from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 11SEP20

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the data set
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values

# Encode categories
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train

# Predict the Test set 