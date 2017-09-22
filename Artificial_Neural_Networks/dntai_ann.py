################################################################## 
# Artificial Neural Network
##################################################################

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

##################################################################
# Part 1 - Data Preprocessing
##################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv');
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder();
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]);
labelencoder_X_2 = LabelEncoder();
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]);

onehotendcoder = OneHotEncoder(categorical_features=[1]);
X = onehotendcoder.fit_transform(X).toarray();
# Remove a contry column for dummy variables
# Because there are three country and only need two country to show
# (0 0) implies that the removed column
X = X[:,1:] 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0);

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##################################################################
# Part 2 - Now let's make the ANN!
##################################################################

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer= 'uniform', activation='relu', input_dim=11));

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


##################################################################
# Part 3 - Making predictions and evaluating the model
##################################################################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print('% Accuracy = ', (tn + tp)*100 / (tn + tp + fn + fp), '%')

# Draw the Confusion Matrix
import seaborn as sn
df_cm = pd.DataFrame(cm, index = [i for i in ['Not Exited','Exited']],
                  columns = [i for i in ['Not Exited','Exited']])
plt.figure(figsize = (5,5))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,fmt='0')
plt.show()

-------------------
# Manage model
-------------------

# Save model
classifier.save('Churn_Modelling.Model.h5')
del classifier

# Load model
from keras.models import load_model

classifier = load_model('Churn_Modelling.Model.h5')
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print('% Accuracy = ', (tn + tp)*100 / (tn + tp + fn + fp), '%')

-------------------
# Using model to predict a customer
-------------------

# Use our ANN model to predict if the customer with the following informations will leave the bank:
# + Geography: France
# + Credit Score: 600
# + Gender: Male
# + Age: 40 years old
# + Tenure: 3 years
# + Balance: $60000
# + Number of Products: 2
# + Does this customer have a credit card ? Yes
# + Is this customer an Active Member: Yes
# + Estimated Salary: $50000
# So should we say goodbye to that customer ?
# Load model
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

classifier = load_model('Churn_Modelling.Model.h5')

sc_X = StandardScaler();
new_prediction = classifier.predict(sc_X.fit_transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

##################################################################
# Part 4 - Evaluating, Improving and Tuning the ANN
##################################################################

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()