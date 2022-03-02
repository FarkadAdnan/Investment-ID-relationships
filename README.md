# -Investment-ID-relationships-
The objective of this notebook is too see if there are relationships between investment IDs using the historical targets through time
 We plot the results
 
Some basic Git commands are:
```
import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mpl_toolkits import mplot3d
from scipy import stats
from pathlib import Path
import pickle
import math
import time
import umap
from sklearn.cluster import KMeans
import ipympl
```
Some basic Git commands are:
```
%%time
n_features = 300
features = [f'f_{i}' for i in range(n_features)]
train = pd.read_pickle('../input/ubiquant-market-prediction-half-precision-pickle/train.pkl')
train.info()
train.head()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3141410 entries, 0 to 3141409
Columns: 303 entries, investment_id to target
dtypes: float16(301), uint16(2)
memory usage: 1.8 GB
CPU times: user 592 ms, sys: 1.96 s, total: 2.56 s
Wall time: 16.7 s
