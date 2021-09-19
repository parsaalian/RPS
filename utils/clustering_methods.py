import numpy as np
from fcmeans import FCM
from sklearn.cluster import KMeans


def FCM_clustering(vectors, assets, config):
    fcm = FCM(n_clusters=config.n_clusters)
    fcm.fit(vectors)
    baskets = { i: [] for i in range(config.n_clusters) }
    memberships = list(map(lambda groups: (-groups).argsort()[:config.max_memberships], fcm.u))
    for i in range(len(memberships)):
        for k in memberships[i]:
            baskets[k].append(assets[i])
    return baskets


def KMEANS_clustering(vectors, assets, config):
    labels = KMeans(
        n_clusters=config.n_clusters,
        random_state=0
    ).fit_predict(vectors)
    baskets = {}
    for i in range(max(labels)):
        baskets[i] = np.array(assets)[np.where(labels == i)[0]]
    return baskets