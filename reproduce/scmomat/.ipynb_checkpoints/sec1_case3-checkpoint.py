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


plt.rcParams["font.size"] = 10

data_dir = data_dir = "/home/yanxh/data/Seurat_demo_data/bm_cite"
input_dir = '/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case3/bm-cite'
out_dir = '/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/scmomat/bm-cite'

print('Reading `mtx` files...')
rna_count_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'rna_mat_count.mtx')).T)
adt_count_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'adt_mat_count.mtx')).T)

rna_names = pd.read_csv(join(data_dir, 'gene_names.csv'))['x'].to_numpy()
adt_names = pd.read_csv(join(data_dir, 'adt_names.csv'))['x'].to_numpy()

cell_names = pd.read_csv(join(data_dir, 'cell_names.csv'))['x'].to_numpy()
meta_data = pd.read_csv(join(data_dir, 'metadata.csv'), index_col=0)
meta_data['batch'] = meta_data.donor.to_numpy()

for p in [0.1, 0.2, 0.4, 0.8]:
    for repeat in range(3):
        new_train_idx = np.load(join(input_dir, f'p={p}_r={repeat}_new_train_idx.npy'))
        test_rna_idx = np.load(join(input_dir, f'p={p}_r={repeat}_test_rna_idx.npy'))
        test_adt_idx = np.load(join(input_dir, f'p={p}_r={repeat}_test_adt_idx.npy'))

        ad_mult_rna = sc.AnnData(rna_count_mat[new_train_idx])
        ad_mult_rna.var_names = rna_names
        sc.pp.highly_variable_genes(ad_mult_rna, flavor='seurat_v3', n_top_genes=5000)
        hvg_idx = np.where(ad_mult_rna.var.highly_variable)[0]
        hvg_names = rna_names[hvg_idx]

        n_batches = 3
        genes = hvg_names
        proteins = adt_names
        feats_name = {"rna": genes, "adt": proteins}

        barcodes = [cell_names[new_train_idx], cell_names[test_rna_idx], cell_names[test_adt_idx]]
        labels = [meta_data.loc[_, 'celltype.l2'].to_list() for _ in barcodes]
        mods = [['multiome']*new_train_idx.size, ['rna']*test_rna_idx.size, ['adt']*test_adt_idx.size]  # mod_lisi=batch_lisi
        batches = [['batch1']*new_train_idx.size, ['batch2']*test_rna_idx.size, ['batch2']*test_adt_idx.size]  # for convenience of eval_bridge

        # READ IN THE COUNT MATRICES
        # scRNA-seq of batch 1
        counts_rna1 = rna_count_mat[new_train_idx][:, hvg_idx]
        counts_rna1 = scmomat.preprocess(counts_rna1.A, modality = "RNA", log = False)
        counts_rna2 = rna_count_mat[test_rna_idx][:, hvg_idx]
        counts_rna2 = scmomat.preprocess(counts_rna2.A, modality = "RNA", log = False)
        counts_rnas = [counts_rna1, counts_rna2, None]

        # scATAC-seq of batch 1
        counts_adt1 = adt_count_mat[new_train_idx]
        counts_adt1 = scmomat.preprocess(counts_adt1.A, modality = "ADT", log = True)
        counts_adt3 = adt_count_mat[test_adt_idx]
        counts_adt3 = scmomat.preprocess(counts_adt3.A, modality = "ADT", log = True)
        counts_adts = [counts_adt1, None, counts_adt3]

        # CREATE THE COUNTS OBJECT
        counts = {"feats_name": feats_name, "nbatches": n_batches, "rna":counts_rnas, "adt": counts_adts}

        #### training
        #------------------------------------------------------------------------------------------------------------------------------------
        # NOTE: Number of latent dimensions, key hyper-parameter, 20~30 works for most of the cases.
        K = 30
        lamb = 0.001 
        T = 2000     # 4000 cause overfit
        interval = 1000
        batch_size = 0.1
        lr = 1e-2
        seed = 0
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        #------------------------------------------------------------------------------------------------------------------------------------

        start_time = time.time()
        model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
        losses = model.train_func(T = T)
        end_time = time.time()
        print("running time: " + str(end_time - start_time))

        # post-processing
        zs = model.extract_cell_factors()

        n_neighbors = 100
        r = None
        resolution = 0.9
        knn_indices, knn_dists = scmomat.calc_post_graph(zs, n_neighbors, njobs = 8, r = r)
        # labels_leiden = scmomat.leiden_cluster(X = None, knn_indices = knn_indices, knn_dists = knn_dists, resolution = resolution)

        ##### evaluation
        ad_mosaic = sc.AnnData(np.vstack(zs), obsm={"X_emb":np.vstack(zs)})
        ad_mosaic.obs['batch'] = np.hstack(batches)
        ad_mosaic.obs['mod']   = np.hstack(mods)
        ad_mosaic.obs['cell_type'] = np.hstack(labels)

        ad_mosaic.obsp['connectivities'] = scmomat.utils._compute_connectivities_umap(
            knn_indices = knn_indices, knn_dists = knn_dists, 
            n_neighbors = 15, set_op_mix_ratio=1.0, local_connectivity=1.0
        )
        ad_mosaic.uns['neighbors'] = {'connectivities_key':'connectivities'}

        # mosaic eval
        r = eval_mosaic(ad_mosaic, label_key='cell_type', batch_keys=['mod'], 
            use_lisi=True, use_rep='X_emb', use_neighbors=True, use_gc=False, use_nmi=False)  # mod-lisi = batch_lisi

        # nmi, ari using nmi search
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb', use_neighbors=True,
            use='nmi', nmi_method='arithmetic')
        print("nmi={:.4f}, ari={:.4f}".format(nmi, ari))

