import numpy as np
np.random.seed(1234)
from additional import ZINB
import time
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
def cluster_scores(latent_space, K, labels_true):
    labels_pred = KMeans(K).fit_predict(latent_space)
    return [silhouette_score(latent_space, labels_true), NMI(labels_true, labels_pred), ARI(labels_true, labels_pred)]


# expression data
data_path = "/home/jzhaoaz/jiazhao/scPI_v2/package/datasets/cortex/"
X_train = np.loadtxt(data_path + "data_train")
X_test = np.loadtxt(data_path + "data_test")
c_train = np.loadtxt(data_path + "label_train")
n_celltype = 7


# repeat fitting ZIFA
n_repeat = 10
for i in range(n_repeat):
	t0 = time.time()

	zinb = ZINB.ZINB(n_components=10, learn_V=True)
	zinb.fit(X_train)

	np.savetxt(str(i) + 'CORTEX_ZINB_time.txt', np.array(time.time() - t0).reshape([-1]))

	latent_train = zinb.transform(X_train)

	scores = cluster_scores(latent_train, n_celltype, c_train)
	np.savetxt(str(i) + 'CORTEX_ZINB_scores.txt', scores)

	# save training results
	np.savetxt(str(i) + 'CORTEX_ZINB_Z.txt', np.array(latent_train))

