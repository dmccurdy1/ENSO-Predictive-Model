import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
idx_10_19_grouping = np.load('idx_10_19_biweekly.npy')


def SVD_cluster_analysis(data_set, PCA_ax = None, output = 'visual', n_comps = 4, enso_tol = 0.5, conserve_shape = False):
    
    if conserve_shape == True:
    
        reshaped_data = np.reshape(data_set, (data_set.shape[0], -1)) 
        imputer = SimpleImputer(strategy='constant',fill_value=0)
        imputed_data = imputer.fit_transform(reshaped_data)
        U, S, VT = np.linalg.svd(imputed_data, full_matrices = 0)

    else:
        
        reshaped_data = np.reshape(data_set, (data_set.shape[0], -1)) 
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(reshaped_data)
        U, S, VT = np.linalg.svd(imputed_data, full_matrices = 0)
    
    SVD_dim = n_comps
    
    if isinstance(PCA_ax,tuple) and np.max(PCA_ax) <= SVD_dim:
        
        if output == 'visual':
            fig, axs = plt.subplots()

            for i in range(imputed_data.shape[0]):
                x = VT[PCA_ax[0],:] @ imputed_data[i,:].T
                y = VT[PCA_ax[1],:] @ imputed_data[i,:].T

                if idx_10_19_grouping[i] < -enso_tol:
                    axs.scatter(x,y,marker = 'o', color = 'blue', s=40)
                elif idx_10_19_grouping[i] > enso_tol:
                    axs.scatter(x,y,marker = 'o', color = 'red', s=40)

                axs.set_xlabel('PC {}'.format(PCA_ax[0]))
                axs.set_ylabel('PC {}'.format(PCA_ax[1]))
        
        elif output == 'numerical':
            
            PCA_list = []
            
            for j in PCA_ax:
                PCA_j = []
                for i in range(imputed_data.shape[0]):
                    x = VT[j,:] @ imputed_data[i,:].T
                    PCA_j.append(x)
                PCA_list.append(PCA_j)
            
            PCA_list = np.array(PCA_list)
            
            return PCA_list
        
    else:
        
        if output == 'visual':
    
            fig, axs = plt.subplots(nrows = SVD_dim, ncols = SVD_dim, figsize = (15,10))

            for j in range(0,SVD_dim):
                for k in range(0,SVD_dim):
                    for i in range(imputed_data.shape[0]):
                        x = VT[j,:] @ imputed_data[i,:].T
                        y = VT[k,:] @ imputed_data[i,:].T

                        if idx_10_19_grouping[i] < -0.5:
                            axs[j,k].scatter(x,y,marker = 'o', color = 'blue', s=40)
                        elif idx_10_19_grouping[i] > 0.5:
                            axs[j,k].scatter(x,y,marker = 'o', color = 'red', s=40)

                        axs[j,k].set_xlabel('PC {}'.format(j))
                        axs[j,k].set_ylabel('PC {}'.format(k))
    
        elif output == 'numerical':
            
            PCA_list = []
    
            for j in range(SVD_dim):
                PCA_j = []
                for i in range(imputed_data.shape[0]):
                    x = VT[j,:] @ imputed_data[i,:].T
                    PCA_j.append(x)
                PCA_list.append(PCA_j)

            PCA_list = np.array(PCA_list)
            
            return PCA_list
                

def PCA_cluster_analysis(data_set, PCA_ax = None, output = 'visual', n_comps = 4, enso_tol = 0.5, conserve_shape = False):

    
    
    if conserve_shape == True:
    
        reshaped_data = np.reshape(data_set, (data_set.shape[0], -1)) 
        imputer = SimpleImputer(strategy='constant',fill_value=0)
        imputed_data = imputer.fit_transform(reshaped_data)
        
    else:
        
        reshaped_data = np.reshape(data_set, (data_set.shape[0], -1)) 
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(reshaped_data)
        
    PCA_dim = n_comps
    
    pca = PCA(n_components=PCA_dim)
    H = pca.fit_transform(imputed_data)
    
    
    
    if isinstance(PCA_ax,tuple) and np.max(PCA_ax) <= PCA_dim:

        if output == 'visual':
            
            fig, axs = plt.subplots()

            for i in range(imputed_data.shape[0]):

                if idx_10_19_grouping[i] < -enso_tol:
                    axs.scatter(H[i,PCA_ax[0]],H[i,PCA_ax[1]],marker = 'o', color = 'blue', s=40)
                elif idx_10_19_grouping[i] > enso_tol:
                    axs.scatter(H[i,PCA_ax[0]],H[i,PCA_ax[1]],marker = 'o', color = 'red', s=40)

                axs.set_xlabel('PC {}'.format(PCA_ax[0]))
                axs.set_ylabel('PC {}'.format(PCA_ax[1])) 
        
        elif output == 'numerical':
            H_list = []
            for i in PCA_ax:
                H_list.append(H[:,i])
            H_list = np.array(H_list)
            
            return H_list

    else:
        
        if output == 'visual':

            fig, axs = plt.subplots(nrows = PCA_dim, ncols = PCA_dim, figsize = (15,10))

            for j in range(0,PCA_dim):
                for k in range(0,PCA_dim):
                    for i in range(imputed_data.shape[0]):

                        if idx_10_19_grouping[i] < -0.5:
                            axs[j,k].scatter(H[i,j],H[i,k],marker = 'o', color = 'blue', s=40)
                        elif idx_10_19_grouping[i] > 0.5:
                            axs[j,k].scatter(H[i,j],H[i,k],marker = 'o', color = 'red', s=40)

                        axs[j,k].set_xlabel('PC {}'.format(j))
                        axs[j,k].set_ylabel('PC {}'.format(k)) 
    
        elif output == 'numerical':

            return H.T


def ICA_cluster_analysis(data_set, PCA_ax = None, output = 'visual', n_comps = 4, enso_tol = 0.5, conserve_shape = False):
    
    if conserve_shape == True:
    
        reshaped_data = np.reshape(data_set, (data_set.shape[0], -1)) 
        imputer = SimpleImputer(strategy='constant',fill_value=0)
        imputed_data = imputer.fit_transform(reshaped_data)
        
    else:
        
        reshaped_data = np.reshape(data_set, (data_set.shape[0], -1)) 
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(reshaped_data)
        
    PCA_dim = n_comps
    
    ica = FastICA(n_components=n_comps, whiten="arbitrary-variance")
    H = ica.fit_transform(imputed_data)
    
    
    
    if isinstance(PCA_ax,tuple) and np.max(PCA_ax) <= PCA_dim:

        if output == 'visual':
            
            fig, axs = plt.subplots()

            for i in range(imputed_data.shape[0]):

                if idx_10_19_grouping[i] < -enso_tol:
                    axs.scatter(H[i,PCA_ax[0]],H[i,PCA_ax[1]],marker = 'o', color = 'blue', s=40)
                elif idx_10_19_grouping[i] > enso_tol:
                    axs.scatter(H[i,PCA_ax[0]],H[i,PCA_ax[1]],marker = 'o', color = 'red', s=40)

                axs.set_xlabel('PC {}'.format(PCA_ax[0]))
                axs.set_ylabel('PC {}'.format(PCA_ax[1])) 
        
        elif output == 'numerical':
            H_list = []
            for i in PCA_ax:
                H_list.append(H[:,i])
            H_list = np.array(H_list)
            
            return H_list

    else:
        
        if output == 'visual':

            fig, axs = plt.subplots(nrows = PCA_dim, ncols = PCA_dim, figsize = (15,10))

            for j in range(0,PCA_dim):
                for k in range(0,PCA_dim):
                    for i in range(imputed_data.shape[0]):

                        if idx_10_19_grouping[i] < -0.5:
                            axs[j,k].scatter(H[i,j],H[i,k],marker = 'o', color = 'blue', s=40)
                        elif idx_10_19_grouping[i] > 0.5:
                            axs[j,k].scatter(H[i,j],H[i,k],marker = 'o', color = 'red', s=40)

                        axs[j,k].set_xlabel('PC {}'.format(j))
                        axs[j,k].set_ylabel('PC {}'.format(k)) 
    
        elif output == 'numerical':

            return H.T