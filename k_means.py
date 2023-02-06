import utilities as utl
import numpy as np
import pandas as pd

if __name__ == '__main__':

    path = ['./datasets/blobs.csv', './datasets/circle.csv', './datasets/elliptical.csv', './datasets/moon.csv']
    
    for i in range(4):
        data = utl.load_data(path[i])
        print(data)
        X = data.iloc[:, :-1]
        X = utl.normalize(np.asanyarray(X))
        WCSS = []
        for K in range(1, 15):
            clusters, means = utl.k_mans(X, K, 300)
            WCSS.append(utl.Wcss(clusters, means, K))
        utl.plot_elbow(WCSS)
