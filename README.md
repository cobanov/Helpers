<p align="center">
  <img src="https://github.com/cobanov/Helpers/blob/master/help.png" width=300>
</p>

<p align="center">This repository contains helper scripts.
<b> - Author: Mert Cobanoglu</b> </p>


[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# Helpers.


### Argument Parser

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

### List Directory

```python
import os
wd = os.getcwd()
os.listdir(wd)
```

### Correlation Matrix

```python
import pandas as pd
import seaborn as sns

corr = d.corr()
sns.heatmap(corr)
```

### Pickle 

```python
import pickle

favorite_color = { "lion": "yellow", "kitty": "red" }
pickle.dump( favorite_color, open( "save.p", "wb" ) )
favorite_color = pickle.load( open( "save.p", "rb" ) )
```

### Timedelta
```python
import datetime

hours_before = datetime.datetime.now() - datetime.timedelta(hours=2)

print(f"Current Time: {datetime.datetime.now().timestamp()}")
print(f"2 Hours Before: {hours_before.timestamp()}")

```

### Logging
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

### Write dataframe with markdown
```python

import pandas as pd

df = pd.read_csv("diabetes.csv")
markdown = df.to_markdown()

text_file = open("sample.txt", "w")
text_file.write(markdown)
text_file.close()
```

### Label Encoding

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

### XGBoost

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

### NaN Percentage

```python
nan_percentage = raw_data.isna().sum() * 100 / len(raw_data)
missing_percentage_df = pd.DataFrame({'column_name': raw_data.columns, 'percent_missing': nan_percentage}).reset_index(drop=True)

percentage_threshold = 20 #define percentage to filter
missing_percentage_df[missing_percentage_df["percent_missing"] < percentage_threshold]
```

### Metrics

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
```
### Classification Report
```python
from sklearnmetrics import classification_report
report = classification_report(y_test, best_preds)
print(report)
```


<!-- TABLE OF CONTENTS -->
## Usefull Tools

#### General
  * [Argument Parser](https://github.com/cobanov/helprepo/blob/master/argumentparser.py)
  * [List Directory](https://github.com/cobanov/helprepo/blob/master/listdir.py)
  * [Flask](https://github.com/cobanov/helprepo/blob/master/flask.py)
#### Tutorials 
* [Pandas](https://github.com/cobanov/helprepo/blob/master/pandas.ipynb)
* [Matplotlib](https://github.com/cobanov/helprepo/blob/master/matplotlib.ipynb)
#### Machine Learning
  * [XGBoost Cross-Validation](https://github.com/cobanov/helprepo/blob/master/xgboost_cv.py)
  * [Hello Tensorflow](https://github.com/cobanov/helprepo/blob/master/deeplearning/tensorflow.py)
  * [Confusion Matrix](https://github.com/cobanov/helprepo/blob/master/deeplearning/confmat.py)
  * [GPU Available](https://github.com/cobanov/helprepo/blob/master/deeplearning/gpu_available.py)
  * [Basic Keras](https://github.com/cobanov/helprepo/blob/master/deeplearning/keras_mnist.py)
  * [More Basic Keras](https://github.com/cobanov/helprepo/blob/master/deeplearning/easykeras.py)


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
