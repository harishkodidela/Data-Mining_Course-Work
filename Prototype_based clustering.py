# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:21:53 2023

@author: Harry
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import plotly.io as pio
import plotly.express as px
SEED = 2019

# Reading the data from dataset file
df = pd.read_csv(r"C:\Users\dheer\OneDrive\Desktop\Harish\DataSet_CredCard.csv")
print (df.shape)

'''Preprocessing the data, dropping time column and dimensionality reduction has already been implemented'''
def data_preprocessing(df):
    df['Cls'] = df['Cls'].str[1].astype('int')
    df['Amt']= (df['Amt'] - df['Amt'].min())/( df['Amt'].max() -  df['Amt'].min())
    df = df.drop('Time',axis=1)
    X = df.drop('Cls',axis=1)
    scaler = MinMaxScaler()
    scaled_cols = X.columns
    X[scaled_cols] = scaler.fit_transform(X)
    y = df['Cls']
    return X,y

'''Plotting the clusters from the data set taken with first two components or first two columns,
   sample size taken were 10000
   providing the scatter cluster plot with x, y labels and with proper title'''
def clustering_plot(df,sample_size=10000):
    df_plot  = df[['V1','V2','cluster']].copy()
    df_plot['cluster'] = df_plot['cluster'].astype(pd.CategoricalDtype())
    df_plot = df_plot.sample(sample_size,random_state=SEED)
    pio.renderers.default='browser'
    fig = px.scatter(df_plot.sample(10000), x='V1', y='V2', color='cluster',title='Clusters plotting for dataset')
    fig.update_layout(autosize=False, width=400, height=400)
    fig.show()
    
'''Plotting the anomalies from the data set taken with first two components or first two columns,
   sample size taken were 10000
   providing the scatter cluster plot with x, y labels and with proper title'''
def anomalies_plot(df,sample_size=10000):
    df_plot  = df[['V1','V2','cluster','anomalies']].copy()
    df_plot['anomalies'] = df_plot['anomalies'].map({0:'Non-Anomaly',1:'Anomaly'})
    df_plot[['anomalies','cluster']] = df_plot[['anomalies','cluster']].astype(pd.CategoricalDtype())
    df_plot = df_plot.sample(sample_size,random_state=SEED)
    pio.renderers.default='browser'
    fig = px.scatter(df_plot, x='V1', y='V2', color='anomalies',title='Anomalies Detected')
    fig.update_layout(autosize=False, width=400, height=400)
    fig.show()
    
X,y= data_preprocessing(df)

'''K-Means Model Based Cluster
   Applying the K means for cluster points from the data set using from sklearn.cluster import KMeans'''
cluster_points = 2
all_components = 2
X_train = X.iloc[:,:cluster_points]
kmeans = KMeans(n_clusters = cluster_points, init='k-means++', random_state=SEED)
kmeans.fit(X_train)
cluster_preds = kmeans.predict(X_train)
df['cluster'] = cluster_preds
df['distance'] = kmeans.transform(X_train).min(axis=1)

clustering_plot(df)

t_hold = df['distance'].mean() + 3*df['distance'].std()
df['anomalies'] = [1 if x > t_hold else 0 for x in df['distance']]
anomalies_plot(df)
print (classification_report(y,df['anomalies']))


'''Doing the iterations
   K-Means Model Based Cluster
   Applying the K means for cluster points from the data set using from sklearn.cluster import KMeans'''
cluster_points = 2
all_components = 31
X_train = X.iloc[:,:all_components]
kmeans = KMeans(n_clusters = cluster_points,init='k-means++', random_state=SEED)
kmeans.fit(X_train)
cluster_preds = kmeans.predict(X_train)
df['cluster'] = cluster_preds
df['distance'] = kmeans.transform(X_train).min(axis=1)

clustering_plot(df)

t_hold = df['distance'].mean() + 3*df['distance'].std()
df['anomalies'] = [1 if x > t_hold else 0 for x in df['distance']]
anomalies_plot(df)
print (classification_report(y,df['anomalies']))

'''A heuristic for counting the number of clusters in a data collection is the elbow approach.
   Plotting the explained variation as a function of the number of clusters, and selecting the
   elbow of the curve as the number of clusters to utilise, constitutes the technique.
'''
def nclusters_train_kmeans(X,n_clusters):
    within_cluster_sum_square = []
    dist= []

    # Iterate through numerous of range of clusters
    for i in range(n_clusters):
        print (f"Running for n_clusters: {i+1}")
        kmeans = KMeans(n_clusters=i+1, init='k-means++',random_state=SEED)
        kmeans.fit(X)
        within_cluster_sum_square.append(kmeans.inertia_)
        dist.append(sum(np.min(cdist(X, kmeans.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
    plt.figure(figsize=(10,5))
    plt.plot(range(1,n_clusters+1), within_cluster_sum_square,'bx-')
    plt.xlabel('Values of clusters K')
    plt.xticks(np.arange(1, n_clusters+1, step=1))
    plt.ylabel('Distortion Values')
    plt.title('Using within cluster sum square drawing elbow curve')
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(range(1,n_clusters+1), within_cluster_sum_square, 'bx-')
    plt.xlabel('Values of K')
    plt.xticks(np.arange(1, n_clusters+1, step=1))
    plt.ylabel('Distortion Values')
    plt.title('Using within cluster sum square drawing elbow curve')
    plt.show()
    
    return within_cluster_sum_square

within_cluster_sum_square = nclusters_train_kmeans(X,20) 

plt.figure(figsize=(10,5))
plt.plot(range(1,20+1), within_cluster_sum_square,'bx-')
plt.xlabel('Values of clusters K')
plt.xticks(np.arange(1, 20+1, step=1))
plt.ylabel('Within cluster sum square')
plt.title('Using within cluster sum square drawing elbow curve')
plt.show()

'''Better Iteration to get best accuracy and precision'''
cluster_points = 5
all_components = 31

X_train = X.iloc[:,:all_components]
kmeans = KMeans(n_clusters = cluster_points, random_state=SEED)
kmeans.fit(X_train)
cluster_preds = kmeans.predict(X_train)
df['cluster'] = cluster_preds
df['distance'] = kmeans.transform(X_train).min(axis=1)

clustering_plot(df)

t_hold = df['distance'].mean() + 6*df['distance'].std()
df['anomalies'] = [1 if x > t_hold else 0 for x in df['distance']]
anomalies_plot(df)

print (classification_report(y,df['anomalies']))