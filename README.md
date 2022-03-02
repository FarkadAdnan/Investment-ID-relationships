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

pivot target so that rows are investment ids and time ids are columns¶
```
inv_piv = train[['investment_id', 'target', 'time_id']].astype(np.float32).pivot(columns='time_id', index='investment_id', 
                                                                                 values='target').fillna(0)
invcols = inv_piv.columns.tolist()
```
We use umap to reduce dimensionality and kmeans to group investment ids. Then we plot the results.
There is some relationships, but this may be due to the distinction between frequent and infrequent investment id's appearing in the dataset

```
%matplotlib inline
pipe = Pipeline([('umap', umap.UMAP(n_components=3, min_dist=0, n_neighbors=10, random_state=21))])
pipe.fit(inv_piv[invcols])
kmeans = KMeans(n_clusters=5)
kmeans.fit(pipe['umap'].embedding_)
# Create the figure
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')

ax.scatter3D(pipe['umap'].embedding_[:, 0], pipe['umap'].embedding_[:, 1], 
           pipe['umap'].embedding_[:, 2], c = kmeans.labels_, s=0.5)
plt.show()
```

![__results___6_0](https://user-images.githubusercontent.com/35774039/156271755-955be203-b79d-4ff7-a3ed-7e8a427dc12a.png)
![__results___6_1](https://user-images.githubusercontent.com/35774039/156271765-ad42d86f-2327-4478-86d8-bda53e49896d.png)
![__results___6_2](https://user-images.githubusercontent.com/35774039/156271766-876de616-5211-44f7-b283-731dc1f811f5.png)

below makes the image interactive using plotly express
```
import plotly.express as px

pipe = Pipeline([('umap', umap.UMAP(n_components=3, min_dist=0, n_neighbors=30, random_state=21))])

pipe.fit(inv_piv[invcols])

kmeans = KMeans(n_clusters=5)
kmeans.fit(pipe['umap'].embedding_)

fig = px.scatter_3d(x=pipe['umap'].embedding_[:, 0], y=pipe['umap'].embedding_[:, 1], z=pipe['umap'].embedding_[:, 2], 
                    color = kmeans.labels_, size=kmeans.labels_, size_max=18, opacity=0.7)

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
```
![newplot](https://user-images.githubusercontent.com/35774039/156272369-9335bfbd-b8f4-49a3-ac70-29b9aa0e3e5a.png)


