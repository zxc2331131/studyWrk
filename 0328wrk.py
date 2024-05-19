from sklearn import datasets
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
X = [0,1,2,3,4,5,6,7,8,9]
X_tsne = manifold.TSNE(n_components=2, random_state=5, init='tandom', verbose=1).fit_transform(X)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]),
             color=plt.cm.Set1(y[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])

plt.show()