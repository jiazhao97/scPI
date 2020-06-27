import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from matplotlib import gridspec
from scipy.spatial import distance_matrix
from additional import ZIFA


# show training and testing loss
def visualize_loss(epoches, losses_train, losses_test):
    fig = plt.figure(figsize=(7, 5)) 
    plt.plot(epoches, losses_train, label="train")
    plt.plot(epoches, losses_test, label="test")
    # plt.axvline(x=500, linestyle ="--", c="red", alpha=0.6)
    # plt.axvline(x=250, linestyle ="--", c="blue", alpha=0.6)
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Objective function")


# clustering scores
def cluster_scores(latent_space, K, labels_true):
    labels_pred = KMeans(K, n_jobs=8, n_init=20).fit_predict(latent_space)
    return [silhouette_score(latent_space, labels_true), NMI(labels_true, labels_pred), ARI(labels_true, labels_pred)]


# imputation error
def imputation_error(X_mean, X, X_zero, i, j, ix):
    all_index = i[ix], j[ix]
    x, y = X_mean[all_index], X[all_index]
    return np.median(np.abs(x - y))


# visualize latent distance
def visualize_distance_cortex(latent, labels):
    celltypes = np.array(['astrocytes_ependymal', 'endothelial-mural', 'interneurons', \
                          'microglia', 'oligodendrocytes', 'pyramidal CA1', 'pyramidal SS'])
    beloved_order = np.array(['interneurons', 'pyramidal SS', 'pyramidal CA1', 'endothelial-mural', \
                          'microglia', 'astrocytes_ependymal', 'oligodendrocytes'])
    mapping = [np.where(beloved_order == x)[0][0] for x in celltypes]

    sorting_labels = [mapping[int(x)] for x in labels]
    order_latent = np.vstack([x for _, x in sorted(zip(sorting_labels, latent), key=lambda pair: pair[0])])
    order_label = np.vstack([x for _, x in sorted(zip(sorting_labels, labels), key=lambda pair: pair[0])])
    distance = distance_matrix(order_latent, order_latent)  
    
    fig = plt.figure(figsize=(5, 5)) 
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 70], height_ratios=[1, 70])
    gs.update(wspace=0.05, hspace=0.05)
    ax0 = plt.subplot(gs[1])
    ax0.imshow(order_label.T, cmap='tab10', interpolation='none', aspect=32)
    ax0.axis('off')
    ax1 = plt.subplot(gs[3], sharex=ax0)
    dis= ax1.imshow(distance, cmap='hot', interpolation='none')
    ax1.axis('off')
    ax2 = plt.subplot(gs[2], sharey=ax1)
    ax2.imshow(order_label, cmap='tab10', interpolation='none', aspect=1/32.)
    ax2.axis('off')


# log-likelihood of FA
def fast_logdet(A):
    """Compute log(det(A)) for A symmetric
    Equivalent to : np.log(nl.det(A)) but more robust.
    """
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld
def FA_logllh(Y, A, mu, W, N, D, K):
    Xr = Y - mu.reshape([1,D])
    Sigma = A @ A.T + np.diag(W.reshape([D,]))
    invSigma = np.linalg.inv(Sigma)
    logllh_v = - 0.5 * (Xr * (np.dot(Xr, invSigma))).sum(axis=1)
    logllh_v -= 0.5 * (D * np.log(2.0*np.pi) - fast_logdet(invSigma))
    logllh_v = np.mean(logllh_v) - np.mean(np.sum(Y, axis=-1))
    return logllh_v


# log-likelihood for ZIFA
def ZIFA_logllh(Y, A, mu, W, lam, N, D, K):
    Y_is_zero = np.array(np.abs(Y) < 1e-6) + 0.0
    Y2 = Y ** 2
    mus = mu
    sigmas = np.sqrt(W)
    decay_coef = lam
    EZ, EZZT, EX, EXZ, EX2, entropy = ZIFA.Estep(Y, A, mus, sigmas, decay_coef)
    EZ2 = np.zeros([N, K])
    for i in range(N):
        EZ2[i,:] = np.diag(EZZT[i,:,:])
    # Q function
    A2 = np.zeros([K, K, D])
    for j in range(D):
        A2[:,:,j] = A[j,:].reshape([K,1]) @ A[j,:].reshape([1,K])
    const = - N*(K+D)/2 * np.log(2*np.pi)
    Q_Z = - 1/2*np.sum(EZ2) - N/2*np.sum(np.log(sigmas ** 2))
    tmp_zero = - decay_coef*EX2 - (EX2 - 2*EX*mus.reshape([1,D]) - 2*np.sum(A.reshape([1,D,K])*EXZ, axis=2)) / 2 / (sigmas.reshape([1,D])**2)
    tmp_nonzero = np.log(1-np.exp(-decay_coef*Y2)+Y_is_zero) - (Y2 - 2*Y*mus.reshape([1,D]) - 2*Y*(EZ@A.T)) / 2 / (sigmas.reshape([1,D])**2)
    tmp = - (EZZT.reshape([N, K*K]) @ A2.reshape([K*K, D]) + 2*mus.reshape([1,D])*(EZ@A.T) + mus.reshape([1,D])**2) / 2 / (sigmas.reshape([1,D])**2)
    Q = const + Q_Z + np.sum(tmp_zero*Y_is_zero) + np.sum(tmp_nonzero*(1-Y_is_zero)) + np.sum(tmp)
    # elbo (elbo = log-likelihood at this time)
    elbo = Q + entropy
    logllh_v = elbo / N - np.mean(np.sum(Y, axis=-1))
    return logllh_v

