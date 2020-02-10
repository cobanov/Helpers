"""

Created on DD-MM-YYYY hh:mm
Author: Mert Cobanoglu  (COB3BU)
        Ezgi Atardag    (ATE6BU)


|==== To-Do ===|
    + getData
    + getDmatrix_train_test
    + Normal Train Model
    + Cross Validation Model
    + Grid Search
    x Predictions
    x Visualization

"""
from time import time 
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import make_scorer

### XGBoost Classification Model
"""
|=============================|
|*** Parameter Definitions ***|
|=============================|
# eta: step size shrinkage used to prevent overfitting. Range is [0,1]
# max_depth: determines how deeply each tree is allowed to grow during any boosting round.
# subsample: percentage of samples used per tree. Low value can lead to underfitting.
# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
# n_estimators: number of trees you want to build.
# objective: determines the loss function to be used like 
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

|=======================|
|***   Evaluation    ***|
|=======================|
If early stopping occurs, the model will have three additional fields: 
bst.best_score, bst.best_iteration and bst.best_ntree_limit. 
Note that xgboost.train() will return a model from the last iteration, not the best one.
"""

### Default Initializers

def_num_boost_round = 10
def_metrics = 'rmse'
def_early_stopping_rounds = 5
def_nfold = 3
def_objective = {'objective' : 'reg:linear'}
def_num_class = 3


# Normal Train Parameters
params_normal = {
                'num_class' : 3, # if objective classification
                # 'eta':0.01,
                # 'gamma' : 0,
                # 'max_depth' : 6,
                # 'min_child_weight' : 1,
                # 'subsample' : 1,
                # 'colsample_bytree' : 1,
                # 'lambda' : 1,
                # 'alpha' : 0,
                'objective' : 'reg:linear'
                }

# Cross Validation Parameters
params_cv = {
                'eta':0.01,
                'gamma' : 0,
                'max_depth' : 6,
                'min_child_weight' : 1,
                'subsample' : 1,
                'colsample_bytree' : 1,
                'lambda' : 1,
                'alpha' : 0,
                'objective' : 'reg:linear',
                'nfold' : 3
                }

# Grid Search Parameters # (5 * 9 * 3 * 4 * 3 * 3) #too much

# params_gs = {   
#                 'n_estimators': range(60, 200, 20),
#                 'max_depth': range(2, 10, 1),
#                 'learning_rate' : [0.001, 0.01, 0.1],
#                 'objective' : '**to_be_defined**',
#                 'gamma': [0.5, 1, 1.5, 2, 5],
#                 'min_child_weight': [1, 5, 10],
#                 'subsample': [0.6, 0.8, 1.0],
#                 'colsample_bytree': np.arange(start, stop, step)
#                 }

params_gs = {   
                'n_estimators': [60, 70],
                'max_depth': [2, 3],
                'learning_rate' : [0.1],
                'gamma': [0.5, 1],
                'min_child_weight': [1, 5],
                'subsample': [0.6, 0.8, 1.0],
                }


def run_model_train(dmatrix_train, dmatrix_test, params=params_normal, num_boost_round=def_num_boost_round, metrics=def_metrics, early_stopping_rounds=def_early_stopping_rounds):
    """ Trains XGBmodel and prints sort of metrics,
        watchlist is using for plotting evaluation so if dmatrix_test already defined easily plots graphics
        in order to observe the model have overfitting problem or not."""

    watchlist = [(dmatrix_test, 'eval'), (dmatrix_train, 'train')]
    evals_result = {}

    model_normal = xgb.train(params=params, dtrain=dmatrix_train, 
                    num_boost_round=num_boost_round,
                    evals=watchlist,
                    evals_result=evals_result
                 )

    predicts = model_normal.predict(dmatrix_test)
    labels =  dmatrix_test.get_label()
    best_preds = np.asarray([np.argmax(line) for line in predicts])

    print("Precision = {}".format(precision_score(labels, best_preds, average='macro')))
    print("Recall = {}".format(recall_score(labels, best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(labels, best_preds)))

    return model_normal, evals_result #returns booster return type: trained booster model



def run_model_cv(dmatrix_train, params=params_cv, show_plot=False, num_boost_round=def_num_boost_round, nfold=def_nfold, metrics=def_metrics, early_stopping_rounds=def_early_stopping_rounds):
    """ Function makes cross validation, this function returns a list(string) different from the above function. """

    model_cv = xgb.cv(params=params, dtrain=dmatrix_train, 
                    num_boost_round=num_boost_round, 
                    nfold=nfold, 
                    early_stopping_rounds=early_stopping_rounds,
                    seed=123
                 )


    if show_plot == True:
        model_cv.plot()

    print(model_cv)
    
    return model_cv #xbg.cv returns evaluation history, return type: list(string)



def run_model_grid_search(X_train, y_train, params_gs, num_class=def_num_class):
    """fgd asdf asd as """
    num_class = num_class
    model_xgb = xgb.XGBRegressor(objective='multi:softprob', num_class=num_class)
    
    model_gs = GridSearchCV(param_grid=params_gs,
                            estimator=model_xgb,
                            n_jobs=-1,
                            verbose=1,
                            refit="accuracy_score")
    
    model_gs.fit(X_train, y_train)
    
    print("Best parameters found: ", model_gs.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(model_gs.best_score_)))

    

    #results = pd.DataFrame(model_gs.cv_results_)
    #results.to_csv("xgb-gs_results.csv", index=False)
    #best_estimator = model_gs.best_estimator_

    return model_gs



# def run_model_predict(model, data_test, objective=param_normal['objective']):
#     """ asdasd """

#     predicts = model.predict(data_test)
#     labels = data_test.get_label()

#     if objective == 'multi:softprob':

#         best_preds = np.asarray([np.argmax(line) for line in predicts])

#         print("Precision = {}".format(precision_score(labels, best_preds, average='macro')))
#         print("Recall = {}".format(recall_score(labels, best_preds, average='macro')))
#         print("Accuracy = {}".format(accuracy_score(labels, best_preds)))
    
#     elif objective == 'reg:linear':
#         pass

#     elif objective == 'reg:logistic':
#         pass

#     elif objective == 'binary:logistic':
#         pass

#     else:
#         print("objective type error!!")


#     return predicts

