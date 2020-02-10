import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import alpha_xgboost as ax
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def_num_boost_round = 10
def_metrics = 'rmse'
def_early_stopping_rounds = 5
def_nfold = 3
def_objective = {'objective' : 'multi:softprob'}
def_num_class = 3


data = pd.read_csv("datasets/iris.csv")
encoder = LabelEncoder()
data["Species"] = encoder.fit_transform(data["Species"])

X_train, X_test, y_train, y_test = ax.getData(data, 
                                            target_col_name="Species", 
                                            test_size=0.2, 
                                            show_shapes=True)

dmatrix_train, dmatrix_test = ax.getDmatrix_train_test(X_train, X_test, y_train, y_test)

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
                'objective' : 'multi:softprob'
                }

model_normal, evals_result = ax.run_model_train(dmatrix_train=dmatrix_train,
                    dmatrix_test=dmatrix_test, params=params_normal)


"""==================CROSS VALIDATION==================
   ===================================================="""


params_cv = {
                # 'eta':0.01,
                # 'gamma' : 0,
                # 'max_depth' : 6,
                # 'min_child_weight' : 1,
                # 'subsample' : 1,
                # 'colsample_bytree' : 1,
                # 'lambda' : 1,
                # 'alpha' : 0,
                "num_class" : 3,
                'objective' : 'multi:softprob',
                'nfold' : 3
                }

model_cv = ax.run_model_cv(dmatrix_train, params=params_cv, 
            show_plot=False, 
            num_boost_round=def_num_boost_round, 
            nfold=def_nfold, metrics=def_metrics, 
            early_stopping_rounds=def_early_stopping_rounds)


             
"""==================GRID SEARCH==================
   ==============================================="""


params_gs = {   
                'n_estimators': [60, 70],
                'max_depth': [2, 3],
                'learning_rate' : [0.1],
                'gamma': [0.5, 1],
                'min_child_weight': [1, 5],
                'subsample': [0.6, 0.8, 1.0],
                }

scorers = {
            'f1_score':make_scorer(f1_score),
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
          }

model_gs = ax.run_model_grid_search(X_train, y_train, params_gs, num_class=3)