# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:27:46 2019

@author: COB3BU
"""
#%%
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%% Import Data
dataframe = pd.read_excel("data1.xlsx")
y_true = dataframe.loc[:,"Result"]
dataframe2 = dataframe.drop("Result",axis=1)

#%% Normalization
from sklearn import preprocessing

x = dataframe2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
#%%

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_true, test_size=0.3, random_state=42)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
#%%

model = Sequential()
model.add(Dense(32, activation="relu", input_shape=[26]))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=300, batch_size=16)

#%%
# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#%%


y_pred = model.predict(X_test)


#%%

decoded_datum = []
decoded_test = []

def decode(datum):
    return np.argmax(datum)

for i in range(y_pred.shape[0]):
    datum = y_pred[i]
    x=decode(y_pred[i])
    decoded_datum.append(x)

for i in range(y_test.shape[0]):
    datum = y_test[i]
    x=decode(y_test[i])
    decoded_test.append(x)

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix   
import seaborn as sns
cm = confusion_matrix(decoded_test, decoded_datum)    
sns.heatmap(cm,annot=True)
















