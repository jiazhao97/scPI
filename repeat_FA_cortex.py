import os
import numpy as np
np.random.seed(1234)
import torch
torch.manual_seed(1234)
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
def cluster_scores(latent_space, K, labels_true):
    labels_pred = KMeans(K, n_jobs=8, n_init=20).fit_predict(latent_space)
    return [silhouette_score(latent_space, labels_true), NMI(labels_true, labels_pred), ARI(labels_true, labels_pred)]

n_celltype = 7
file_name = "CORTEX_FA"

# expression data
data_path = "/home/jzhaoaz/jiazhao/scPI_v2/CORTEX_data/CORTEX_data/"
expression_train = np.log(np.loadtxt(data_path + "data_train", dtype='float32') + 1)
# expression_test = np.log(np.loadtxt(data_path + "data_test", dtype='float32') + 1)
c_train = np.loadtxt(data_path + "label_train")

n_repeat = 10
for i in range(n_repeat):
	t0 = time.time()

	alg = FactorAnalysis(n_components=10)
	alg.fit(expression_train)
	latent = alg.transform(expression_train)
	np.savetxt(str(i) + file_name + '_time.txt', np.array(time.time()-t0).reshape([-1]))

	scores = cluster_scores(latent, n_celltype, c_train)
	np.savetxt(str(i) + file_name + '_scores.txt', scores)

	if (i==0):
		np.savetxt(str(i) + file_name + '_Z.txt', latent)

