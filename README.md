[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# Helpers


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






<!-- TABLE OF CONTENTS -->
## Usefull Tools

#### General
  * [Argument Parser](https://github.com/cobanov/helprepo/blob/master/argumentparser.py)
  * [List Directory](https://github.com/cobanov/helprepo/blob/master/listdir.py)
  * [Flask](https://github.com/cobanov/helprepo/blob/master/flask.py)
#### Tutorials 
* [Pandas](https://github.com/cobanov/helprepo/blob/master/pandas.ipynb)
* [Matplotlib](https://github.com/cobanov/helprepo/blob/master/matplotlib.ipynb)
#### Deep Learning
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
