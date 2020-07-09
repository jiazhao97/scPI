import numpy as np
np.random.seed(1234)
from additional import ZIFA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
def cluster_scores(latent_space, K, labels_true):
    labels_pred = KMeans(K).fit_predict(latent_space)
    return [silhouette_score(latent_space, labels_true), NMI(labels_true, labels_pred), ARI(labels_true, labels_pred)]


# expression data
data_path = "/home/jzhaoaz/jiazhao/scPI_v2/CORTEX_data/CORTEX_data/"
expression_train = np.log(np.loadtxt(data_path + "data_train", dtype='float32') + 1)
expression_test = np.log(np.loadtxt(data_path + "data_test", dtype='float32') + 1)
c_train = np.loadtxt(data_path + "label_train")


# repeat fitting ZIFA
result, result_test = ZIFA.fitModel(Y=expression_train, K=10, Y_test=expression_test)
# save training results
np.savetxt('CORTEX_ZIFA_Z.txt', result["latent"])
np.savetxt('CORTEX_ZIFA_A.txt', result['A'])
np.savetxt('CORTEX_ZIFA_mus.txt', result['mus'])
np.savetxt('CORTEX_ZIFA_W.txt', np.array(result['sigmas'])**2)
np.savetxt('CORTEX_ZIFA_lam.txt', np.array(result['decay_coef']).reshape([-1]))
np.savetxt('CORTEX_ZIFA_time.txt', np.array(result['run_time']).reshape([-1]))
np.savetxt('CORTEX_ZIFA_logllh.txt', np.array(result['logllh']).reshape([-1]))
# save testing results
np.savetxt('CORTEX_ZIFA_latent_test.txt', result_test["latent"])
np.savetxt('CORTEX_ZIFA_logllh_test.txt', np.array(result_test["logllh"]).reshape([-1]))