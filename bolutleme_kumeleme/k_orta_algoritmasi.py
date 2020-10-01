#kutuphaneler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri kumesinin okunmasi

veriler = pd.read_csv("musteriler.csv")

#verilerin dilimlenmesi

X = veriler.iloc[:,3:].values

#k orta algoritmasi

from sklearn.cluster import KMeans

kmeans = KMeans (n_clusters = 3, init = "k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init="k-means++", random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)