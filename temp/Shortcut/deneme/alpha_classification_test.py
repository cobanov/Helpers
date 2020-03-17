from alpha_main import alpha_data as ad 
from alpha_main import alpha_classification as ac
import pandas as pd 

data = pd.read_csv("datasets/iris.csv")
data.drop(labels=["Id"], axis=1, inplace=True)

print(data.head())

X_train, X_test, y_train, y_test = ad.getData(data, "Species", 0.2)
print(y_train)

dmatrix_train, dmatrix_test = ad.getDmatrix_train_test(X_train, X_test, y_train, y_test)

#ac.run_model_train(dmatrix_train=dmatrix_train, dmatrix_test=dmatrix_test)

#ac.run_model_cv(dmatrix_train=dmatrix_train, show_plot=True)

#ac.run_model_grid_search(X_train, y_train)