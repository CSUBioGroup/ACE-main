import sys, os
import h5py
import numpy as np
from umap import UMAP
import datetime
import torch
import matplotlib.pyplot as plt
import pandas as pd  
import scanpy as sc
import scipy.sparse as sps
import scipy.io as sio
# import scipy.sparse as sp
from os.path import join

import scmomat 
import sys

plt.rcParams["font.size"] = 10

data_dir = '/home/cb213/local/cache3/yxh/Data/seurat4-CITE-reference'
data_dir2 = '/home/yanxh/gitrepo/multi-omics-matching/tmp_outputs/time_complx/inputs'

print('Reading `mtx` files...')
_path = '/home/sda1/yanxh/data/seurat-CITE-reference/cite.h5'
with h5py.File(_path, 'r') as f:
    cell_names = np.array(f['cellID'], dtype='S32').astype('str')
    rna_count_data = sps.csc_matrix(
            (np.array(f['RNA.count.data'], dtype=np.float32), 
             np.array(f['RNA.count.indices'], dtype=np.int32),
             np.array(f['RNA.count.indptr'], dtype=np.int32)
            ), 
            shape = np.array(f['RNA.shape'], dtype=np.int32)
    ).tocsc().astype(np.float32).T#.toarray()
    rna_names = np.array(f['rna_names'], dtype='S32').astype('str')
    
    # adt_norm_data = np.array(f['adt_norm_data'], dtype=np.float32)
    adt_count_data = np.array(f['adt_count_data'], dtype=np.float32)
    protein_names = np.array(f['protein_names'], dtype='S32').astype('str')
    
    meta_data = pd.DataFrame(
        dict(
            donor=np.array(f['donor'], dtype='S32').astype('str'),
            celltype_l1=np.array(f['celltype.l1'], dtype='S32').astype('str'),
            celltype_l2=np.array(f['celltype.l2'], dtype='S32').astype('str'),
            celltype_l3=np.array(f['celltype.l3'], dtype='S32').astype('str'),
            Phase=np.array(f['Phase'], dtype='S32').astype('str'),
            X_index=np.array(f['X_index'], dtype='S32').astype('str'),
            lane=np.array(f['lane'], dtype='S32').astype('str'),
            time=np.array(f['time'], dtype='S32').astype('str')
        ),
        index=cell_names
    )

ad_cite = sc.AnnData(rna_count_data, obs=meta_data.loc[cell_names])
ad_cite.var_names = rna_names
sc.pp.highly_variable_genes(ad_cite, flavor='seurat_v3', n_top_genes=5000)
hvg_names = ad_cite.var.query('highly_variable').index.to_numpy()
ad_cite = ad_cite[:, hvg_names].copy()
ad_cite.obs['batch'] = ad_cite.obs.donor + '-' + ad_cite.obs.time

ad_cite.obsm['adt'] = adt_count_data

# post-processing step takes too much memory, rate>=0.8 => memory overflow
for rate in [0.01, 0.1, 0.2, 0.4, 0.8, 1.0]:   # [0.01, 0.1, 0.2, 0.4, 0.8, 1.0]
    smp_names = pd.read_csv(join(data_dir2, f'names_{rate}.csv'))['0'].values
    n_smp = len(smp_names)
    n_interval = n_smp // 3
    ad_cite_subset = ad_cite[smp_names].copy()
    batch = ad_cite_subset.obs.batch.to_numpy()

    genes = hvg_names
    proteins = protein_names
    feats_name = {"rna": genes, "adt": proteins}

    counts_rnas, counts_adts = [], []
    # mult-part 
    mult_batch_list = batch[:n_interval]
    for bi in np.unique(mult_batch_list):
        b_msk = mult_batch_list == bi
        counts_rnas.append(
            scmomat.preprocess(ad_cite_subset[:n_interval][b_msk].X.A, modality='RNA', log=False)
        )
        counts_adts.append(
            scmomat.preprocess(ad_cite_subset[:n_interval][b_msk].obsm['adt'], modality='ADT', log=True)
        )
    # rna-part
    rna_batch_list = batch[n_interval:int(2*n_interval)]
    for bi in np.unique(rna_batch_list):
        b_msk = rna_batch_list == bi
        counts_rnas.append(
            scmomat.preprocess(ad_cite_subset[n_interval:int(2*n_interval)][b_msk].X.A, modality='RNA', log=False)
        )
        counts_adts.append(None)
    # adt-part
    adt_batch_list = batch[int(2*n_interval):int(3*n_interval)]
    for bi in np.unique(adt_batch_list):
        b_msk = adt_batch_list == bi
        counts_rnas.append(None)
        counts_adts.append(
            scmomat.preprocess(ad_cite_subset[int(2*n_interval):int(3*n_interval)][b_msk].obsm['adt'], modality='ADT', log=True)
        )
    n_batches = len(counts_rnas)

    # CREATE THE COUNTS OBJECT
    counts = {"feats_name": feats_name, "nbatches": n_batches, "rna":counts_rnas, "adt": counts_adts}

    #### training
    #------------------------------------------------------------------------------------------------------------------------------------
    # NOTE: Number of latent dimensions, key hyper-parameter, 20~30 works for most of the cases.
    K = 30
    lamb = 0.001 
    T = 4000     # 4000 cause overfit
    interval = 1000
    batch_size = 0.1
    lr = 1e-2
    seed = 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #------------------------------------------------------------------------------------------------------------------------------------

    start_time = datetime.datetime.now()
    model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
    losses = model.train_func(T = T)

    # post-processing
    zs = model.extract_cell_factors()

    n_neighbors = 100
    r = None
    resolution = 0.9
    knn_indices, knn_dists = scmomat.calc_post_graph(zs, n_neighbors, njobs = 8, r = r)
    # labels_leiden = scmomat.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)

    _ = scmomat.utils._compute_connectivities_umap(
        knn_indices = knn_indices, knn_dists = knn_dists, 
        n_neighbors = 15, set_op_mix_ratio=1.0, local_connectivity=1.0
    )
    
    end_time = datetime.datetime.now()
    print('===============================================')
    print(f'Rate {rate}')
    print("running time: ", (end_time - start_time).total_seconds())
    print('===============================================')