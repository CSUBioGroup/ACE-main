from sklearn.neighbors import NearestNeighbors
import scipy
import numpy as np
import pandas as pd
from itertools import combinations

# faster version
import torch
import gc
import math
def batch_gpu_pairdist(emb1, emb2, batch_size=1024):
    n_batch = math.ceil(emb2.shape[0] / batch_size)
    emb2_gpu = torch.FloatTensor(emb2).cuda()
    emb2_gpu = emb2_gpu / torch.linalg.norm(emb2_gpu, ord=2, dim=1, keepdim=True)
    
    st = 0
    dist = []
    for i in range(n_batch):
        bsz = min(batch_size, emb1.shape[0] - i*batch_size)
        emb1_batch_gpu = torch.FloatTensor(emb1[st:st+bsz]).cuda()
        emb1_batch_gpu /= torch.linalg.norm(emb1_batch_gpu, ord=2, dim=1, keepdim=True)
        
        _ = -emb1_batch_gpu @ emb2_gpu.T  # 0-similarity => dist
        dist.append(_.cpu().numpy())
        st = st+bsz
        
        del emb1_batch_gpu
        torch.cuda.empty_cache()
        gc.collect()
    
    del emb2_gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    dist = np.vstack(dist)
    return dist

def eval_FOSCTTM(adata1, adata2, use_rep='X_emb', return_dist=False):
    dist = batch_gpu_pairdist(adata1.obsm[use_rep], adata2.obsm[use_rep], batch_size=2048)
    foscttm_x = (dist < dist.diagonal().reshape(-1, 1)).mean(axis=1)
    foscttm_y = (dist < dist.diagonal()).mean(axis=0)
    foscttm = (foscttm_x+foscttm_y).mean()/2

    return foscttm, dist

def eval_FOSCTTM_above2(adatas, use_rep='X_emb', mod_key='mod', return_dist=False):
    n_omics = len(adatas)
    combs = combinations(np.arange(n_omics), 2)
    fosc_dict = {}
    for c in combs:
        fosc, _ = eval_FOSCTTM(adatas[c[0]], adatas[c[1]], use_rep=use_rep)
        fosc_dict['FOSCTTM: '+adatas[c[0]].uns[mod_key]+'-'+adatas[c[1]].uns[mod_key]] = fosc # FOSCTTM: gex-adt

    return fosc_dict

def eval_ACC(scores, label_x=None, label_y=None, K=5, use_rep='X_emb'):
    # knn classifier
    x_top_nn = np.argpartition(-scores, kth=K, axis=1)[:, :K]
    y_top_nn = np.argpartition(-scores, kth=K, axis=0)[:K, :]
    
    top_pred_x = label_y[x_top_nn]
    top_pred_y = label_x[y_top_nn]

    acc_x = np.any(top_pred_x == label_x.reshape(-1, 1), axis=1).mean()
    acc_y = np.any(top_pred_y == label_y, axis=0).mean()
    acc = (acc_x.mean() + acc_y.mean())/2

    if K>1:
        knn_pred_x = np.array(list(map(lambda x: scipy.stats.mode(x)[0][0], top_pred_x)))
        knn_pred_y = np.array(list(map(lambda x: scipy.stats.mode(x)[0][0], top_pred_y.T)))
        knn_pred_x_acc = (label_x == knn_pred_x).mean()
        knn_pred_y_acc = (label_y == knn_pred_y).mean()
        knn_acc = (knn_pred_x_acc + knn_pred_y_acc) / 2
        return acc, knn_pred_x_acc, knn_pred_y_acc

    return acc

def snn_scores(
        x, y, k=1
):
    '''
        return: matching score matrix
    '''

    # print(f'{k} neighbors to consider during matching')

    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    ky = k or min(round(0.01 * y.shape[0]), 1000)   
    nny = NearestNeighbors(n_neighbors=ky).fit(y)
    x2y = nny.kneighbors_graph(x)
    y2y = nny.kneighbors_graph(y)

    kx = k or min(round(0.01 * x.shape[0]), 1000)
    nnx = NearestNeighbors(n_neighbors=kx).fit(x)
    y2x = nnx.kneighbors_graph(y)
    x2x = nnx.kneighbors_graph(x)

    x2y_intersection = x2y @ y2y.T
    y2x_intersection = y2x @ x2x.T
    jaccard = x2y_intersection + y2x_intersection.T
    jaccard.data = jaccard.data / (2 * kx + 2 * ky - jaccard.data)
    matching_matrix = jaccard.multiply(1 / jaccard.sum(axis=1)).tocsr()
    return matching_matrix

def eval_matching_score(
        mod1, mod2, split_by='batch', k=1, use_rep='X'
):  
    '''
        return: scipy.sparse.csr_matrix
    '''

    mod1_splits = set(mod1.obs[split_by])
    mod2_splits = set(mod2.obs[split_by])
    splits = mod1_splits | mod2_splits
    
    matching_matrices, mod1_obs_names, mod2_obs_names = [], [], []
    for split in splits:
        mod1_split = mod1[mod1.obs[split_by] == split]
        mod2_split = mod2[mod2.obs[split_by] == split]
        mod1_obs_names.append(mod1_split.obs_names)
        mod2_obs_names.append(mod2_split.obs_names)
        
        matching_matrices.append(
            snn_scores(mod1_split.X, mod2_split.X, k)
            if use_rep=='X' else
            snn_scores(mod1_split.obsm[use_rep], mod2_split.obsm[use_rep], k)
        )
        
    mod1_obs_names = pd.Index(np.concatenate(mod1_obs_names))
    mod2_obs_names = pd.Index(np.concatenate(mod2_obs_names))
    combined_matrix = scipy.sparse.block_diag(matching_matrices, format="csr")
    score_matrix = combined_matrix[
        mod1_obs_names.get_indexer(mod1.obs_names), :
    ][
        :, mod2_obs_names.get_indexer(mod2.obs_names)
    ]

    score = (score_matrix.diagonal() / score_matrix.sum(axis=1).A1).mean()
    return score

def eval_matching_score_above2(mods, split_by='batch', mod_key='mod', k=1, use_rep='X'):
    n_omics = len(mods)
    combs = combinations(np.arange(n_omics), 2)
    score_dict = {}
    for c in combs:
        score = eval_matching_score(mods[c[0]], mods[c[1]], split_by=split_by, k=k, use_rep=use_rep)
        score_dict['Matching score: '+mods[c[0]].uns[mod_key] +'-'+ mods[c[1]].uns[mod_key]] = score

    return score_dict

