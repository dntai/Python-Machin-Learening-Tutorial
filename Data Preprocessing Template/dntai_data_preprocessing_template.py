# Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv');
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X[:,1:3]);
X[:,1:3] = imputer.transform(X[:,1:3]);

# Update and Save dataset
dataset.iloc[:, :-1] = X
dataset.to_csv('Data_out.csv');

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder();
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]);
onehotendcoder = OneHotEncoder(categorical_features=[0]);
X = onehotendcoder.fit_transform(X).toarray();

labelencoder_Y = LabelEncoder();
y = labelencoder_Y.fit_transform(y);


float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter});
X

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0);

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
