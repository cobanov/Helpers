from time import time 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def getData(df, target_col_name, test_size, show_shapes=True):
    """ Get data from 'DataFrame', should defined col_name in order to seperation,
        function returns 4 parameters which are train and test data
        show_shapes shows which shapes that they are """
    
    
    if df[target_col_name].dtype == "object":
        encoder = LabelEncoder()
        df[target_col_name] = encoder.fit_transform(df[target_col_name])

    data_without_target = df.drop(columns=target_col_name)
    X_train, X_test, y_train, y_test = train_test_split(data_without_target, df[target_col_name], test_size=test_size, random_state=123)
    
    if show_shapes == True:
        for datas in [X_train, X_test, y_train, y_test]:
            print(datas.shape)       

    return X_train, X_test, y_train, y_test


def getDmatrix_train_test(X_train, X_test, y_train, y_test):
    """ This function converts data to DMatrix format, they are using in XGBModels like train or cv."""

    dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
    dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)

    return dmatrix_train, dmatrix_test
