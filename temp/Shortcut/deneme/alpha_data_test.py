from alpha_main import alpha_data 
import pandas as pd 

data = pd.read_csv("datasets/iris.csv")
data.drop(labels=["Id"], axis=1, inplace=True)

print(data.head())

X_train, X_test, y_train, y_test = alpha_data.getData(data, "Species", 0.2)
print(y_train)