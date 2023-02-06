import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def normalize(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


def load_data(path):
    data = pd.read_csv(path)
    return data


def k_mans(X, K, max_itr):
    samples, features = X.shape
    clusters = [[] for i in range(K)]
    random_sample = np.random.choice(samples, K, replace=False)
    means = [X[idx] for idx in random_sample]
    for i in range(max_itr):
        clusters = create_clusters(means, X, K)
        means_old = means
        means = get_means(clusters, K, features)
        if np.array_equal(means_old, means):
            break
    plot(clusters, means)
    return clusters, means


def create_clusters(means, X, K):
    clusters = [[] for _ in range(K)]
    for sample in X:
        dist = [math.dist(sample, i) for i in means]
        centroid_idx = np.argmin(dist)
        clusters[centroid_idx].append(sample)
    return clusters     


def closest_centroid(sample, means):
    dist = [math.dist(sample, i) for i in means]
    idx = np.argmin(dist)
    return idx    


def get_means(clusters, K, features):
    means = np.zeros((K, features))
    for idx, cluster in enumerate(clusters):
        cluster_mean = np.mean(np.array(cluster), axis=0)
        means[idx] = cluster_mean
    return means  


def Wcss(clusters, means, K):
    wcss = 0
    for i in range(K):
        tmp = clusters[i]
        for j in tmp:
            wcss += (np.linalg.norm(j - means[i])) ** 2
    return wcss      


def getlabel(clusters):
    x = []
    y = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            x.append(j)
            y.append(i)
    y = np.array(y)
    x = np.array(x)
    return x, y


def plot(clusters, means):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, index in enumerate(clusters):
        point = np.array(index).T
        ax.scatter(*point)
    for point in means:
        ax.scatter(*point, marker="x", color="black", linewidth=2)
    plt.show()    


def plot_elbow(WCSS):
    K_array = np.arange(1, 15, 1)
    plt.plot(K_array, WCSS)
    plt.xlabel('Number of Clusters')
    plt.ylabel('within-cluster sums of squares (WCSS)')
    plt.title('Elbow method to determine optimum number of clusters')
    plt.show()


def calc_mean_cov(x, n_components):
    split = np.array_split(x, n_components)
    means = [np.mean(s, axis=0) for s in split]
    covariances = [np.cov(s, rowvar=False) for s in split]
    return means, covariances


def e_step(X, phi, mu, sigma, n_components):
    pdf = []
    for i in range(n_components):
        pdf.append(multivariate_normal.pdf(X, mu[i], sigma[i], allow_singular=True) * phi[i])
    sum_w = np.sum(pdf, axis=0)
    w = pdf / sum_w
    return w


def m_step(X, w):
    n_components = len(w)
    phi, mu, sigma = [], [], []
    for i in range(n_components):
        phi.append(np.sum(w[i]) / len(X))
        mu.append(np.dot(w[i], X) / np.sum(w[i]))
        temp = []
        for j in range(len(X)):
            temp.append(w[i][j] * np.dot((X[j] - mu[i]).reshape(2, 1), (X[j] - mu[i]).reshape(1, 2)))
        sigma.append(np.sum(temp, axis=0) / np.sum(w[i]))
    return phi, mu, sigma
