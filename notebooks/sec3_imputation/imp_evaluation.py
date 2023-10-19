import os
import gc
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scanpy as sc
import anndata as ad
import scipy.io as sio

from os.path import join
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

def binarize(X):
    X[X>0] = 1.
    return X

def eval_aucRmse_AlongPeak(X, Y):
    aucs, rmses = [], []
    for pi in range(X.shape[1]):
        aucs.append(roc_auc_score(X[:, pi], Y[:, pi]))
        rmses.append(
            np.sqrt(np.mean((X[:, pi] - Y[:, pi])**2))
        )
    return aucs, rmses

import scipy
import copy
import gc
def pearson_mat(X0, Y0):
    X, Y = copy.deepcopy(X0), copy.deepcopy(Y0)
    X = (X - X.mean(axis=0))
    X /= (scipy.linalg.norm(X, axis=0, ord=2) + 1e-12)
    Y = (Y - Y.mean(axis=0))
    Y /= (scipy.linalg.norm(Y, axis=0, ord=2) + 1e-12)
    res = (X * Y).sum(axis=0)
    del X, Y
    gc.collect()
    return res

def pearson_mat_axis1(X0, Y0):
    X, Y = copy.deepcopy(X0), copy.deepcopy(Y0)
    X = (X - X.mean(axis=1, keepdims=True))
    X /= (scipy.linalg.norm(X, axis=1, ord=2, keepdims=True) + 1e-12)
    Y = (Y - Y.mean(axis=1, keepdims=True))
    Y /= (scipy.linalg.norm(Y, axis=1, ord=2, keepdims=True) + 1e-12)
    res = (X * Y).sum(axis=1)
    del X, Y
    gc.collect()
    return res

def eval_PearSpear_AlongGene(X, Y):
    pears = pearson_mat(X, Y)
    spears = []
    for gi in range(X.shape[1]):
        spears.append(scipy.stats.spearmanr(X[:, gi], Y[:, gi])[0])
    return pears, spears

def eval_PearSpear_AlongCell(X, Y):
    pear_alongcell = pearson_mat_axis1(X, Y)
    spear_alongcell = []
    for ci in range(X.shape[0]):
        spear_alongcell.append(scipy.stats.spearmanr(X[ci, ], Y[ci, ])[0])
    return pear_alongcell, spear_alongcell
        
def eval_imputation_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    rmse = np.sqrt(np.mean((x - y)**2))
    print(f"Found rmse {rmse:.4f}")
    return pearson_r, spearman_corr, rmse

def eval_imputation_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    rmse = np.sqrt(np.mean((x - y)**2))
    print(f"Found rmse {rmse:.4f}")
    return pearson_r, spearman_corr, rmse

from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
def get_umap(ada, scale=False, norm=False):
    if norm:
        ada.X = normalize(ada.X, norm='l2', axis=1)
    if scale:
        sc.pp.scale(ada)
    if ada.shape[1] > 50:
        use_rep = 'X_pca'
        sc.pp.pca(ada, n_comps=50)
    else:
        use_rep = 'X'
    sc.pp.neighbors(ada, n_neighbors=15, use_rep=use_rep)
    sc.tl.umap(ada)
    return ada

def get_rna_umap(adata):
    X = adata.X.A if sps.issparse(adata.X) else adata.X
    u, s, vh = randomized_svd(X, 50, n_iter=15, random_state=0)
    X_lsi = X @ vh.T / s
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm['X_lsi'] = X_lsi
    sc.pp.neighbors(adata, use_rep='X_lsi', n_neighbors=15)
    sc.tl.umap(adata)
    return adata

def get_adt_umap(adata):
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata)
    return adata

def get_atac_umap(adata):
    X = adata.X.copy()
    if scipy.sparse.issparse(X):
        idf = X.shape[0] / (1e-12 + X.sum(axis=0).A1)
        tf = X.multiply(1 / (1e-12 + X.sum(axis=1)))
        X = tf.multiply(idf)
        X = X.multiply(1e4 / (1e-12 + X.sum(axis=1)))
    else:
        idf = X.shape[0] / (1e-12 + X.sum(axis=0))
        tf = X / (1e-12 + X.sum(axis=1, keepdims=True))
        X = tf * idf
        X = X * (1e4 / (1e-12 + X.sum(axis=1, keepdims=True)))
    X = np.log1p(X)
    u, s, vh = randomized_svd(X, 50, n_iter=15, random_state=0)
    X_lsi = X @ vh.T / s
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm['X_lsi'] = X_lsi
    sc.pp.neighbors(adata, use_rep='X_lsi', n_neighbors=15)
    sc.tl.umap(adata)
    return adata

def save_umap(adata, use_rep, colors, is_title='', legend_loc=None, save_dir=None, prefix_name=None):
    for c in colors:
        sc.pl.embedding(adata, basis=use_rep, color=[c], legend_fontsize=4, legend_loc=legend_loc,
              frameon=False, legend_fontoutline=2, show=False, title=is_title) # cmap=reds, vmin=0.00001, 
        plt.savefig(f"{save_dir}/{prefix_name}_{c}_legend={legend_loc}.jpg", bbox_inches="tight", dpi=300)