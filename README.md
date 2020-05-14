<p align="center">
  <img src="https://github.com/cobanov/Helpers/blob/master/img/help.png" width=300>
</p>

<p align="center">This repository contains helper scripts.
<b> - Author: Mert Cobanoglu</b> </p>


[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# Helpers.

## Contents
* [Python](#python)
* [Data Manipulation](#data-manipulation)
* [Statistics](#statistics)
* [Visualization](#visualization)
* [Machine Learning](#machine-learning)

## Python
#### Argument Parser

```python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--isim","-i")
parser.add_argument("--soyisim","-s")
parser.add_argument("--no","-n")

veri = parser.parse_args()

print("isim {}".format(veri.isim))
print("soyisim {}".format(veri.soyisim))
print("no {}".format(veri.no))
```


#### List Directory

```python
path = r"C:\Users\path"
filenames = os.listdir(path)

for i in filenames:
    dirs = os.path.join(path, i)
    print(dirs)
```

#### Select files with extensions

```python
import glob, os
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".ipynb"):
            print(os.path.join(root, file))
```

#### Pickle 

```python
import pickle

favorite_color = { "lion": "yellow", "kitty": "red" }
pickle.dump( favorite_color, open( "save.p", "wb" ) )
favorite_color = pickle.load( open( "save.p", "rb" ) )
```

#### Timedelta
```python
import datetime

hours_before = datetime.datetime.now() - datetime.timedelta(hours=2)

print(f"Current Time: {datetime.datetime.now().timestamp()}")
print(f"2 Hours Before: {hours_before.timestamp()}")

``` 

#### Logging
```python
import logging

logging.basicConfig(filename='test.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def add(x, y):
    """Add Function"""
    return x +

num_1 = 20
num_2 = 10

add_result = add(num_1, num_2)
logging.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))

```

## Statistics

#### Correlation Matrix

```python
import pandas as pd
import seaborn as sns

corr = d.corr()
sns.heatmap(corr)
```

#### NaN Percentage (This is not a clever way also useless, nevertheless i won't remove it.)

```python
nan_percentage = raw_data.isna().sum() * 100 / len(raw_data)
missing_percentage_df = pd.DataFrame({'column_name': raw_data.columns, 'percent_missing': nan_percentage}).reset_index(drop=True)

percentage_threshold = 20 #define percentage to filter
missing_percentage_df[missing_percentage_df["percent_missing"] < percentage_threshold]
```

#### Write dataframe with markdown
```python

import pandas as pd

df = pd.read_csv("diabetes.csv")
markdown = df.to_markdown()

text_file = open("sample.txt", "w")
text_file.write(markdown)
text_file.close()
```

#### Label Encoding

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import pandas as pd

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"] 
data = pd.read_csv("iris.data", names=cols)

#Label Encoding

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(data["class"])

#One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(sparse=False)
targets = targets.reshape(150, 1)
oneho = oh_encoder.fit_transform(targets)

for cols in data.columns:
    data[cols] = label_encoder.fit_transform(data[cols])
```

#### Determine how many extra columns would be created


```python
# Select the object (string) columns
mask = data.dtypes == np.object
categorical_cols = data.columns[mask]

num_ohc_cols = (data[categorical_cols]
                .apply(lambda x: x.nunique())
                .sort_values(ascending=False))
                
# No need to encode if there is only one value
small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]

# Number of one-hot columns is one less than the number of categories
small_num_ohc_cols -= 1

# This is 215 columns, assuming the original ones are dropped. 
# This is quite a few extra columns!
small_num_ohc_cols.sum()                             
```

## Machine Learning
[More on  machine learning repo](https://github.com/cobanov/Helpers/tree/master/machine_learning)

#### Get notifications when the model has finished

```python
# Model Kütüphaneleri
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier

# Bildirim Kütüphaneleri
from win10toast import ToastNotifier
import time

# # Toplam süreyi hesaplamak ve bunu bildirimde görmek iyi olabilir.
start = time.process_time()
model = RandomForestClassifier(n_estimators=700).fit(X_train, y_train)
duration = time.process_time() - start

# # Model tahminlerini alalım
preds = model.predict(X_test)

# # Metriklerimizi alalım
acc = accuracy_score(y_test, preds))
prec = (precision_score(y_test, preds))

# Bildirim objemizi oluşturalım
toaster = ToastNotifier()
toaster.show_toast("Eğitim bitti",
                   f"{acc}, {model_precision}, Süre: {duration}",
                   icon_path=None,
                   duration=5,
                   threaded=True)
```

#### Show plots

```python
for name in data.columns[:20]: #Limit columns to plot on data 
    plt.figure(figsize=(30,10)) #Change figure size
    sns.scatterplot(x=data[name], y=range(0, data[name].shape[0])) #Make scatter plots
    plt.show() #Show every plot on every iterations in order to not to wait for all
```

#### XGBoost

```python
import xgboost as xgboost
import pandas as pd

churn_data = pd.read_csv("classification_data.csv")

churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:, -1],
                            label=churn_data.month_5_still_here)

params = {"objective":"binary:logistic", max_depth=4}

cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=4,
                    num_boost_round=10, metrics="error", as_pandas=True)
```


#### Metrics

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
```
#### Classification Report
```python
from sklearnmetrics import classification_report
report = classification_report(y_test, best_preds)
print(report)
```

## Visualization
[More on visualizaton repo](https://github.com/cobanov/Helpers/tree/master/visualization)
```python
def dact_dist(dataset, high_corrs, class_col):
    
    """
    :dataset: pandas dataframe
    :values: columns to visualize
    :class_col: classes
    """
    
    labels = dataset[class_col].value_counts().index.to_list()
    for col_name in high_corrs:
        fig, ax = plt.subplots(figsize=(30,10))
        for label in labels: 
            sns.distplot(dataset[col_name][dataset[class_col]==label], ax=ax)
            ax.legend(labels)
        plt.show()
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = read_csv("./train.csv")

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
correlation_heatmap(train)
```
```python

categories = ["A", "B", "C"]
plt.figure(figsize=(30,5))

for cat in categories:
    g = sns.kdeplot(data_70[data['Feat1']==cat]["Feat2"],shade=True, bw=.01)
    g.set_xlim(59,65)
```

### Virtual Env and Pip Best Practices

```python
python -m venv [directory] #Create venv
myvenv/bin/activate.bin #activate, for windows just click
pip install simplejson # regular installing via python
pip install --upgrade pip # this is also commonly known
pip freeze > requirements.txt # this is best :), for venv it creates requ.txt 
pip install -r requirements.txt # easy way to install all dependencies
deactivate # deactivate env :)

```


<!-- CONTACT -->
## Contact

Mert Cobanoglu - [Linkedin](https://www.linkedin.com/in/mertcobanoglu/) - mertcobanov@gmail.com


<!-- MARKDOWN LINKS & IMAGES -->
[build-shield]: https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square
[contributors-shield]: https://img.shields.io/badge/contributors-1-orange.svg?style=flat-square
[license-shield]: https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square
[license-url]: https://choosealicense.com/licenses/mit
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: https://raw.githubusercontent.com/othneildrew/Best-README-Template/master/screenshot.png
