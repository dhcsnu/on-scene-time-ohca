#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class FAMD():
    def __init__(self, con_var, cat_var):
        self.n_components = None
        self.con_var = con_var
        self.cat_var = cat_var
        self.var_length = len(con_var) + len(cat_var)
    
    def fit(self, data, n_components):
        data = data.copy()
        self.n_components = n_components
        
        data[self.con_var] = data[self.con_var].apply(lambda x: x.astype(float))
        data[self.cat_var] = data[self.cat_var].apply(lambda x: x.astype(int))

        con_data = self.calculate_zscore(data[self.con_var], self.con_var)
        cat_data = self.normalize_and_center_columns(data[self.cat_var], self.cat_var)
        self.processed_data = pd.concat([con_data, cat_data], axis=1)

        self.pca = PCA(n_components=self.n_components, random_state=0)
        self.pca.fit(self.processed_data)
        
        print("Explained variance: %.1f" %(np.sum(self.pca.explained_variance_ratio_) * 100))
        return self.pca   
    
    def transform(self, data):
        data = data.copy()
        data[self.con_var] = data[self.con_var].apply(lambda x: x.astype(float))
        data[self.cat_var] = data[self.cat_var].apply(lambda x: x.astype(int))
        
        con_data = self.calculate_zscore_test(data[self.con_var], self.con_var)
        cat_data = self.normalize_and_center_columns_test(data[self.cat_var], self.cat_var)
        processed_data = pd.concat([con_data, cat_data], axis=1)
        return self.pca.transform(processed_data)
    
    def fit_transform(self, data, n_components):
        self.fit(data, n_components)
        return self.pca.transform(self.processed_data)
    
    def plot_explained_variance(self, data):
        data = data.copy()
        data[self.con_var] = data[self.con_var].apply(lambda x: x.astype(float))
        data[self.cat_var] = data[self.cat_var].apply(lambda x: x.astype(int))

        con_data = self.calculate_zscore(data[self.con_var], self.con_var)
        cat_data = self.normalize_and_center_columns(data[self.cat_var], self.cat_var)
        processed_data = pd.concat([con_data, cat_data], axis=1)
    
        x, y = [], []

        for i in range(1, self.var_length+1):
            pca = PCA(n_components=i, random_state=0)
            pca.fit(processed_data)
            x.append(i)
            y.append(np.sum(pca.explained_variance_ratio_) * 100)

        plt.figure(figsize=(6,3))
        plt.plot(x,y, marker='o', markersize=4, color='black')
        plt.xlabel("Number of dimensions")
        plt.xticks(np.arange(0, self.var_length+2, 2))
        plt.ylabel("Explained variance (%)")
        plt.show()

    def calculate_zscore(self, data, columns):
        self.con_mean = data[columns].mean()
        self.con_std = data[columns].std(ddof=0)
        data[columns] = data[columns].apply(lambda x: (x - self.con_mean[x.name]) / self.con_std[x.name])
        return data
    
    def calculate_zscore_test(self, data, columns):
        data[columns] = data[columns].apply(lambda x: (x - self.con_mean[x.name]) / self.con_std[x.name])
        return data

    def normalize_and_center_columns(self, data, columns):
        length = len(data)
        self.weight_dict = {col: math.sqrt(data[col].sum() / length) for col in columns}
        self.cat_mean = {}

        for col in columns:
            weight = self.weight_dict[col]
            data.loc[:, col] = data[col] / weight
            self.cat_mean[col] = data[col].mean()
            data.loc[:, col] = data[col] - self.cat_mean[col]

        return data
    
    def normalize_and_center_columns_test(self, data, columns):
        for col in columns:
            weight = self.weight_dict[col]
            data.loc[:, col] = data[col] / weight
            mean = self.cat_mean[col]
            data.loc[:, col] = data[col] - mean

        return data

