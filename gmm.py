import utilities as utl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal


if __name__ == '__main__':
    path = ['./datasets/blobs.csv', './datasets/circle.csv', './datasets/elliptical.csv',
            './datasets/moon.csv', './datasets/tsnv.csv']
    components = [5]
    iterations = 200
    for i in range(1):
        data = utl.load_data(path[i])
        if i == 4:
            x = data.iloc[:, :]
        else:
            x = data.iloc[:, :-1]
        x = utl.normalize(x).values
        # plt.scatter(x[:, 0], x[:, 1], marker='.')
        # plt.show()
        for n_component in components:
            phi = [1 / n_component for _ in range(n_component)]
            mu, sigma = [], []
            clusters = np.array_split(x, n_component)
            for cluster in clusters:
                mu.append(np.mean(cluster, axis=0))
                sigma.append(np.cov(cluster, rowvar=False))
            x1 = np.linspace(0, 1, 100)
            x2 = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x1, x2)
            Z0 = multivariate_normal(mu[0], sigma[0])
            Z1 = multivariate_normal(mu[1], sigma[1])
            Z2 = multivariate_normal(mu[2], sigma[2])
            Z3 = multivariate_normal(mu[3], sigma[3])
            Z4 = multivariate_normal(mu[4], sigma[4])
            # Z5 = multivariate_normal(mu[5], sigma[5])
            # Z6 = multivariate_normal(mu[6], sigma[6])
            # Z7 = multivariate_normal(mu[7], sigma[7])
            # Z8 = multivariate_normal(mu[8], sigma[8])
            # Z9 = multivariate_normal(mu[9], sigma[9])
            pos = np.empty(X.shape + (2,))  # a new array of given shape and type, without initializing entries
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            plt.figure(figsize=(10, 10))  # creating the figure and assigning the size
            plt.scatter(x[:, 0], x[:, 1], marker='o')
            plt.contour(X, Y, Z0.pdf(pos), alpha=0.5)
            plt.contour(X, Y, Z1.pdf(pos), alpha=0.5)
            plt.contour(X, Y, Z2.pdf(pos), alpha=0.5)
            plt.contour(X, Y, Z3.pdf(pos), alpha=0.5)
            plt.contour(X, Y, Z4.pdf(pos), alpha=0.5)
            # plt.contour(X, Y, Z5.pdf(pos), alpha=0.5)
            # plt.contour(X, Y, Z6.pdf(pos), alpha=0.5)
            # plt.contour(X, Y, Z7.pdf(pos), alpha=0.5)
            # plt.contour(X, Y, Z8.pdf(pos), alpha=0.5)
            # plt.contour(X, Y, Z9.pdf(pos), alpha=0.5)
            plt.axis('equal')  # making both the axis equal
            plt.xlabel('X-Axis', fontsize=16)  # X-Axis
            plt.ylabel('Y-Axis', fontsize=16)  # Y-Axis
            plt.title('Initial State', fontsize=22)  # Title of the plot
            plt.grid()  # displaying gridlines
            plt.show()
            iterations = 100
            lis1 = [phi, mu, sigma]
            for mn in range(0, iterations):
                ev = utl.e_step(x, phi, mu, sigma, n_component)
                phi, mu, sigma = utl.m_step(x, ev)
                if mn % 10 == 0:
                    x1 = np.linspace(0, 1, 100)
                    x2 = np.linspace(0, 1, 100)
                    X, Y = np.meshgrid(x1, x2)
                    Z0 = multivariate_normal(mu[0], sigma[0])
                    Z1 = multivariate_normal(mu[1], sigma[1])
                    Z2 = multivariate_normal(mu[2], sigma[2])
                    Z3 = multivariate_normal(mu[3], sigma[3])
                    Z4 = multivariate_normal(mu[4], sigma[4])
                    # Z5 = multivariate_normal(mu[5], sigma[5])
                    # Z6 = multivariate_normal(mu[6], sigma[6])
                    # Z7 = multivariate_normal(mu[7], sigma[7])
                    # Z8 = multivariate_normal(mu[8], sigma[8])
                    # Z9 = multivariate_normal(mu[9], sigma[9])
                    pos = np.empty(X.shape + (2,))  # a new array of given shape and type, without initializing entries
                    pos[:, :, 0] = X
                    pos[:, :, 1] = Y
                    plt.figure(figsize=(10, 10))  # creating the figure and assigning the size
                    plt.scatter(x[:, 0], x[:, 1], marker='o')
                    plt.contour(X, Y, Z0.pdf(pos), alpha=0.5)
                    plt.contour(X, Y, Z1.pdf(pos), alpha=0.5)
                    plt.contour(X, Y, Z2.pdf(pos), alpha=0.5)
                    plt.contour(X, Y, Z3.pdf(pos), alpha=0.5)
                    plt.contour(X, Y, Z4.pdf(pos), alpha=0.5)
                    # plt.contour(X, Y, Z5.pdf(pos), alpha=0.5)
                    # plt.contour(X, Y, Z6.pdf(pos), alpha=0.5)
                    # plt.contour(X, Y, Z7.pdf(pos), alpha=0.5)
                    # plt.contour(X, Y, Z8.pdf(pos), alpha=0.5)
                    # plt.contour(X, Y, Z9.pdf(pos), alpha=0.5)
                    plt.axis('equal')  # making both the axis equal
                    plt.xlabel('X-Axis', fontsize=16)  # X-Axis
                    plt.ylabel('Y-Axis', fontsize=16)  # Y-Axis
                    plt.title('Initial State', fontsize=22)  # Title of the plot
                    plt.grid()  # displaying gridlines
                    plt.show()
            print(phi)
