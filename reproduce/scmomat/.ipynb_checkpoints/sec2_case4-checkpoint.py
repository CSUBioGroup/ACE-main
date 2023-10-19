import sys, os
import numpy as np
from umap import UMAP
import time
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
sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/ACE/reproduce/evaluation')
from evaluation import eval_mosaic, eval_bridge, print_results, eval_lisi, eval_clustering
from evaluation import eval_bridge_above2

plt.rcParams["font.size"] = 10

# Path to the data directory
data_dir = "/home/sda1/yanxh/data/DOGMA"
print('Reading `mtx` files...')
rna_count_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'RNA/rna_mat_count.mtx')).T)
adt_count_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'ADT/adt_mat_count.mtx')).T)
atac_count_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'ATAC/atac_mat_count.mtx')).T)

rna_names = pd.read_csv(join(data_dir, 'RNA/hvg_names.csv'))['VariableFeatures(data_ref)'].to_numpy()
adt_names = pd.read_csv(join(data_dir, 'ADT/adt_names.csv'))['VariableFeatures(data_ref)'].to_numpy()
atac_names = pd.read_csv(join(data_dir, 'ATAC/hvp_names.csv'))['VariableFeatures(data_ref)'].to_numpy()

cell_names = pd.read_csv(join(data_dir, 'cell_names.csv'))['x'].to_numpy()
meta_data = pd.read_csv(join(data_dir, 'metadata.csv'), index_col=0)
meta_data = meta_data[['stim', 'predicted.celltype.l1', 'predicted.celltype.l2']].copy()

# train_idx = np.where((meta_data.stim=='Control').to_numpy())[0]
test_idx  = np.where((meta_data.stim=='Stim').to_numpy())[0]

for p in [0.1, 0.2, 0.4, 0.8]:
    for repeat in range(3):
        new_train_idx = np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/dogma/p={p}_r={repeat}_new_train_idx.npy')
        test_rna_idx  = np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/dogma/p={p}_r={repeat}_test_rna_idx.npy')
        test_adt_idx  = np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/dogma/p={p}_r={repeat}_test_adt_idx.npy')
        test_atac_idx  = np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/dogma/p={p}_r={repeat}_test_atac_idx.npy')

        ## taking highly variable peaks
        hvp_idx = np.argsort(atac_count_mat[new_train_idx].sum(axis=0).A1)[-20000:]
        hvp_names = atac_names[hvp_idx]

        n_batches = 4
        genes = rna_names
        proteins = adt_names
        peaks = hvp_names
        feats_name = {"rna": genes, "adt": proteins, 'atac':peaks}

        barcodes = [cell_names[new_train_idx], cell_names[test_rna_idx], cell_names[test_adt_idx], cell_names[test_atac_idx]]
        labels = [meta_data.loc[_, 'predicted.celltype.l1'].to_list() for _ in barcodes]
        mods = [['multiome']*new_train_idx.size, ['rna']*test_rna_idx.size, ['adt']*test_adt_idx.size, ['atac']*test_atac_idx.size]
        batches = [['control']*new_train_idx.size, ['stim']*test_rna_idx.size, ['stim']*test_adt_idx.size, ['stim']*test_atac_idx.size]

        # READ IN THE COUNT MATRICES
        # scRNA-seq of batches
        counts_rna1 = rna_count_mat[new_train_idx]
        counts_rna1 = scmomat.preprocess(counts_rna1.A, modality = "RNA", log = False)
        counts_rna2 = rna_count_mat[test_rna_idx]
        counts_rna2 = scmomat.preprocess(counts_rna2.A, modality = "RNA", log = False)
        counts_rnas = [counts_rna1, counts_rna2, None, None]

        # adt of batches
        counts_adt1 = adt_count_mat[new_train_idx]
        counts_adt1 = scmomat.preprocess(counts_adt1.A, modality = "ADT")
        counts_adt3 = adt_count_mat[test_adt_idx]
        counts_adt3 = scmomat.preprocess(counts_adt3.A, modality = "ADT")
        counts_adts = [counts_adt1, None, counts_adt3, None]

        # atac of batches
        counts_atac1 = atac_count_mat[new_train_idx][:, hvp_idx]
        counts_atac1 = scmomat.preprocess(counts_atac1.A, modality = "ATAC")
        counts_atac4 = atac_count_mat[test_atac_idx][:, hvp_idx]
        counts_atac4 = scmomat.preprocess(counts_atac4.A, modality = "ATAC")
        counts_atacs = [counts_atac1, None, None, counts_atac4]

        # CREATE THE COUNTS OBJECT
        counts = {"feats_name": feats_name, "nbatches": n_batches, "rna":counts_rnas, "adt": counts_adts, 'atac':counts_atacs}

        # training
        #------------------------------------------------------------------------------------------------------------------------------------
        # NOTE: Number of latent dimensions, key hyper-parameter, 20~30 works for most of the cases.
        K = 30
        #------------------------------------------------------------------------------------------------------------------------------------
        # NOTE: Here we list other parameters in the function for illustration purpose, most of these parameters are set as default value.
        # weight on regularization term, default value
        lamb = 0.001 
        # number of total iterations, default value
        T = 4000
        # print the result after each ``interval'' iterations, default value
        interval = 1000
        # batch size for each iteraction, default value
        batch_size = 0.1
        # learning rate, default value
        lr = 1e-2
        # random seed, default value
        seed = 0
        # running device, can be CPU or GPU
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        #------------------------------------------------------------------------------------------------------------------------------------

        start_time = time.time()
        model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
        losses = model.train_func(T = T)
        end_time = time.time()
        print("running time: " + str(end_time - start_time))

        zs = model.extract_cell_factors()

        n_neighbors = 100
        r = None
        resolution = 0.9
        knn_indices, knn_dists = scmomat.calc_post_graph(zs, n_neighbors, njobs = 8, r = r)

        ad_mosaic = sc.AnnData(np.vstack(zs), obsm={"X_emb":np.vstack(zs)})
        ad_mosaic.obs['batch'] = np.hstack(batches)
        ad_mosaic.obs['mod']   = np.hstack(mods)
        ad_mosaic.obs['cell_type'] = np.hstack(labels)
        ad_mosaic.obs['mod-batch'] = (ad_mosaic.obs['mod'] + '-' + ad_mosaic.obs.batch).to_numpy()

        ad_mosaic.obsp['connectivities'] = scmomat.utils._compute_connectivities_umap(
            knn_indices = knn_indices, knn_dists = knn_dists, 
            n_neighbors = 15, set_op_mix_ratio=1.0, local_connectivity=1.0
        )
        ad_mosaic.uns['neighbors'] = {'connectivities_key':'connectivities'}

        print('=========================')
        print(f'p={p}, repeat={repeat}')
        print('=========================')
        # mosaic eval
        r = eval_mosaic(ad_mosaic, label_key='cell_type', batch_keys=['mod-batch'], use_rep='X_emb', 
            use_lisi=True, use_neighbors=True, use_nmi=False, use_gc=False)  # mod-lisi = batch_lisi

        # nmi, ari using nmi search
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb', use_neighbors=True,
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))
