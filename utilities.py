import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def Normalize(x):
    x = (x - x.min() ) / (x.max() - x.min())
    return x


def load_data(path):
    data= pd.read_csv(path)
    return data

def KMeans(X,K,max_itr):
    
    samples,features = X.shape
    clusters = [[] for i in range(K)]
    random_sample = np.random.choice(samples,K, replace=False)
    means = [X[idx] for idx in random_sample]

    for i in range(max_itr):
        clusters = create_clusters(means,X,K)
        
        means_old = means
        means=get_means(clusters,K,features)

        if np.array_equal(means_old,means) == True :
            break
    plot(clusters,means)
    return clusters,means        

def create_clusters(means,X,K):
    # assign the samples to the closest means
    clusters = [[] for _ in range(K)]
    for sample in X:
        dist= [math.dist(sample, i) for i in means]
        centroid_idx=np.argmin(dist)
        clusters[centroid_idx].append(sample)
    return clusters     

def closest_centroid(sample,means):
    dist = [math.dist(sample, i) for i in means]
    idx = np.argmin(dist)
    return idx    

def get_means(clusters,K,features): 
    means= np.zeros((K,features))
    for idx, cluster in enumerate(clusters):
        cluster_mean = np.mean(np.array(cluster), axis=0)
        means[idx] = cluster_mean
    return means  

def Wcss(clusters,means,K):
    wcss=0
    for i in range(K):
        tmp=clusters[i]
        for j in tmp:
            wcss+=(np.linalg.norm(j - means[i]))**2
       
    return wcss      

def getlabel(clusters):
    x=[]
    y=[]
    for i in range(len(clusters)):
        for j in clusters[i]:
            x.append(j)
            y.append(i)
    y=np.array(y)
    x=np.array(x)
    return x,y


def plot(clusters,means):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, index in enumerate(clusters):
        point = np.array(index).T
        ax.scatter(*point)

    for point in means:
        ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()    

def plot_elbow(WCSS):
    K_array=np.arange(1,15,1)
    plt.plot(K_array,WCSS)
    plt.xlabel('Number of Clusters')
    plt.ylabel('within-cluster sums of squares (WCSS)')
    plt.title('Elbow method to determine optimum number of clusters')
    plt.show() 


# data=load_data()
# X = data.iloc[:,:].values
# y = data.iloc[:, -1].values
      
# clusters = len(np.unique(y))
# print(clusters)
# WCSS=[]
# for K in range(1,15):
#     clusters,means=KMeans(X,K,300)
#     WCSS.append(Wcss(clusters,means))

# plot_elbow(WCSS)
 
