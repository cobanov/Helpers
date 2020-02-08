"""

Created on DD-MM-YYYY hh:mm
Author: Mert Cobanoglu  (COB3BU)
        Ezgi Atardag    (ATE6BU)


|==== To-Do ===|
     x Classification(RDF, XGBoost(Mert)
     x Cross validation
     # Data separation
     x Hyper parameter tuning

"""
from time import time 
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

### XGBoost Classification Model
"""
|=============================|
|*** Parameter Definitions ***|
|=============================|

# learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
# max_depth: determines how deeply each tree is allowed to grow during any boosting round.
# subsample: percentage of samples used per tree. Low value can lead to underfitting.
# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
# n_estimators: number of trees you want to build.

#objective: determines the loss function to be used like 
    ** 'reg:linear'      **     for regression problems, 
    ** 'reg:logistic'    **     for classification problems with only decision
    ** 'binary:logistic' **     for classification problems with probability
    ** 'multi:softprob'  **     for classification problems with multi-class probability
    --https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
|=======================|
|*** Reg. Parameters ***|
|=======================|

# gamma: controls whether a given node will split based on the expected reduction in loss after the split. 
A higher value leads to fewer splits. Supported only for tree-based learners.
# alpha: L1 regularization on leaf weights. A large value leads to more regularization.
# lambda: L2 regularization on leaf weights and is smoother than L1 regularization.
"""


def_num_boost_round = 10
def_metrics = "rmse"
def_early_stopping_rounds = 5


# Normal Train Parameters
params_normal = {
                "learning_rate":0.01,
                "max_depth" : 3,
                "subsample" : 1,
                "colsample_bytree" : 0.3,
                "n_estimators" : 10,
                "objective" : 1
                }

# Cross Validation Parameters
params_cv = {
                "learning_rate":0.01,
                "max_depth" : 3,
                "subsample" : 1,
                "colsample_bytree" : 0.3,
                "n_estimators" : 10,
                "objective" : 1,
                "nfold" : 3
                }

# Grid Search Parameters
params_gs = {
                'colsample_bytree': np.arange(start, stop, step),
                'learning_rate' : [0.001, 0.01, 0.1]
                'n_estimators': [50],
                'max_depth': [2, 5]
                }

# Random Search Parameters
params_rs = {
                'colsample_bytree': np.arange(start, stop, step),
                'n_estimators': [50],
                'max_depth': [2, 5]
                }


def getData(df, target_col_name, test_size, show_shapes=True):
    """ """
    data_without_target = df.drop(columns=target_col_name)
    X_train, X_test, y_train, y_test = train_test_split(data_without_target, df[target_col_name], test_size=test_size, random_state=123)
    
    if show_shapes == True:
        for datas in [X_train, X_test, y_train, y_test]:
            print(datas.shape)        

    return X_train, X_test, y_train, y_test


def getDmatrix_train_test(X_train, X_test, y_train, y_test):

    data_dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
    data_dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)

    return data_dmatrix_train, data_dmatrix_test



# def getDmatrix(df, target_col_name):
#     """Make data to DMatrix form"""

#     data_without_target = df.drop(columns=target_col_name)
#     data_dmatrix = xgb.DMatrix(data=data_without_target, label=df[target_col_name])
    
#     print(f"Number of columns: {data_dmatrix.num_col}")
#     print(f"Number of rows: {data_dmatrix.num_row}")
#     print(f"Feature names: {data_dmatrix.feature_names}")

#     return data_dmatrix



def run_model_train(data_dmatrix_train, params=params_normal, num_boost_round=def_num_boost_round, metrics=def_metrics, early_stopping_rounds=def_early_stopping_rounds):
    """ """
    model_normal = xgb.train(params=params, dtrain=data_dmatrix_train, 
                    num_boost_round=num_boost_round,
                    metrics=metrics, 
                    early_stopping_rounds=early_stopping_rounds,
                    seed=123
                 )
    return model_normal #returns booster return type: trained booster model


def run_model_train_eval(data_dmatrix_train, data_dmatrix_test, params=params_normal, num_boost_round=def_num_boost_round, metrics=def_metrics, early_stopping_rounds=def_early_stopping_rounds):
    """ """
    watchlist = [(data_dmatrix_test, 'eval'), (data_dmatrix_train, 'train')]

    model_normal_eval = xgb.train(params=params, dtrain=data_dmatrix_train, 
                    num_boost_round=num_boost_round,
                    metrics=metrics,
                    evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds,
                    seed=123
                 )

    return model_normal_eval #returns booster return type: trained booster model



def run_model_cv(data_dmatrix, params=params_cv, show_plot=False, num_boost_round=def_num_boost_round, nfold=def_nfold, metrics=def_metrics, early_stopping_rounds=def_early_stopping_rounds):
    """ """

    model_cv = xgb.cv(params=params, dtrain=data_matrix, 
                    num_boost_round=num_boost_round, 
                    nfold=nfold,
                    metrics=metrics, 
                    early_stopping_rounds=early_stopping_rounds,
                    seed=123
                 )


    if show_plot == True:
        model_cv.plot()

    print(model_cv)
    
    return model_cv #xbg.cv returns evaluation history, return type: list(string)




def run_model_grid_search(X_train, y_train, params_gs):
    """fgd asdf asd as """

    regressor_grid = xgb.XGBRegressor()
    grid_mse = GridSearchCV(param_grid=params_gs,
                            estimator=regressor_grid,
                            scoring="neg_mean_squared_error",
                            cv=4, 
                            verbose=1)
    grid_mse.fit(X_train, y_train)

    print("Best parameters found: ", grid_mse.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
    
    return grid_mse



def run_model_random_search(X_train, y_train, params, params_gs):
    """ dajsdas """

    regressor_random = xgb.XGBRegressor()

    randomized_mse = RandomizedSearchCV(param_distributions=gbm_param_grid,
                                        estimator=regressor_random, scoring="neg_mean_squared_error",
                                        n_iter=5, cv=4, verbose=1)
    randomized_mse.fit(X_train, y_train)

    print("Best parameters found: ", randomized_mse.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))



def run_model_predict(model, data_test, y_test, objective=param_normal['objective']):
    """ asdasd """

    predicts = model.predict(data_test)

    if objective == 'multi:softprob':

        best_preds = np.asarray([np.argmax(line) for line in preds])
        print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
        print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
        print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
    
    elif objective == 'reg:linear':
        pass

    elif objective == 'reg:logistic':
        pass

    elif objective == 'binary:logistic':
        pass

    else:
        print("objective type error!!")


    return predicts
