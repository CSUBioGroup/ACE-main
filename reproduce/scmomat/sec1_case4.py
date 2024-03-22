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
sys.path.insert(0, '.')
from evaluation import eval_mosaic, eval_specific_mod, eval_bridge, print_results, eval_asw, eval_lisi, eval_clustering


plt.rcParams["font.size"] = 10

dat_dir = "/media/asus/data16t/xuhua/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/"
data_dir = os.path.join(dat_dir, "match_modality/openproblems_bmmc_cite_phase2_mod2")
input_dir ='/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/cite'

print('Reading `h5ad` files...')
input_train_mod1_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod1.h5ad'))
input_train_mod2_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod2.h5ad'))
input_train_sol = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_sol.h5ad'))
input_test_mod1_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_mod1.h5ad'))
input_test_mod2_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_mod2.h5ad'))
input_test_sol = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_sol.h5ad'))

print("Unscrambling training cells...")
ord = input_train_sol.X.tocsr().indices
train_mod2_ord = ord.copy()
if "pairing_ix" in input_train_sol.uns:
    assert np.all(ord == np.argsort(input_train_sol.uns["pairing_ix"]))
input_train_mod2_0 = input_train_mod2_0[ord, :].copy()
input_train_mod2_0.obs_names = input_train_mod1_0.obs_names

# reorder testing cells
test_ord = input_test_sol.X.tocsr().indices
assert (test_ord == np.argsort(input_test_sol.uns['pairing_ix'])).all()

meta_dir = "/media/asus/data16t/xuhua/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/"
df_meta = pd.read_csv(os.path.join(meta_dir, 'output/datasets/cite_meta.csv'), index_col=0)
input_test_mod1_0.obs['cell_type'] = df_meta.loc[input_test_mod1_0.obs_names, 'cell_type'].to_numpy()
input_test_mod2_0.obs['cell_type'] = input_test_mod1_0.obs.cell_type[np.argsort(test_ord)].to_numpy()
assert np.all(input_test_mod1_0.obs.cell_type.to_numpy() == input_test_mod2_0.obs.cell_type.to_numpy()[test_ord])

NMI1, ARI1, MOD_LISI1, BATCH_LISI1, FOSCTTM1, MS1 = [], [], [], [], [], []
NMI2, ARI2, MOD_LISI2, BATCH_LISI2, FOSCTTM2, MS2 = [], [], [], [], [], []
INDS = []
for p in [0.1, 0.2, 0.4, 0.8]:
    for repeat in range(3):
        if not os.path.exists(join(input_dir, f'p={p}_r={repeat}_new_train_idx.npy')):
            break
        
        new_train_idx = np.load(join(input_dir, f'p={p}_r={repeat}_new_train_idx.npy'))
        test_rna_idx = np.load(join(input_dir, f'p={p}_r={repeat}_test_rna_idx.npy'))
        test_adt_idx = np.load(join(input_dir, f'p={p}_r={repeat}_test_adt_idx.npy'))

        input_train_mod1 = input_train_mod1_0[new_train_idx, ].copy()
        input_train_mod2 = input_train_mod2_0[new_train_idx, ].copy()
        input_test_mod1 = input_test_mod1_0[test_adt_idx].copy()
        input_test_mod2 = input_test_mod2_0[test_rna_idx].copy()

        # hvg
        sc.pp.highly_variable_genes(input_train_mod2, layer='counts', flavor='seurat_v3', n_top_genes=5000, batch_key='batch')
        hvg_names = input_train_mod2.var.query('highly_variable').index.to_numpy()

        genes = hvg_names
        proteins = input_train_mod1.var_names.to_numpy()
        feats_name = {"rna": genes, "adt": proteins}

        # split batches
        train_batches = input_train_mod1.obs.batch.to_numpy()
        test1_batches = input_test_mod1.obs.batch.to_numpy()
        test2_batches = input_test_mod2.obs.batch.to_numpy()

        train_batches_barcodes, train_batches_list, train_mods = [], [], []
        for bi in np.unique(train_batches):
            b_mask = train_batches == bi
            train_batches_barcodes.append(input_train_mod1.obs_names[b_mask].to_numpy())
            train_batches_list.append([bi]*(b_mask.sum()))
            train_mods.append(['multiome']*(b_mask.sum()))
        test1_batches_barcodes, test1_batches_list, test1_mods = [], [], []
        for bi in np.unique(test1_batches):
            b_mask = test1_batches == bi
            test1_batches_barcodes.append(input_test_mod1.obs_names[b_mask].to_numpy())
            test1_batches_list.append([bi]*b_mask.sum())
            test1_mods.append(['adt']*(b_mask.sum()))
        test2_batches_barcodes, test2_batches_list, test2_mods = [], [], []
        for bi in np.unique(test2_batches):
            b_mask = test2_batches == bi
            test2_batches_barcodes.append(input_test_mod2.obs_names[b_mask].to_numpy())
            test2_batches_list.append([bi]*b_mask.sum())
            test2_mods.append(['rna']*(b_mask.sum()))
            
        barcodes = [*train_batches_barcodes, *test1_batches_barcodes, *test2_batches_barcodes]
        batches = [*train_batches_list, *test1_batches_list, *test2_batches_list]
        mods = [*train_mods, *test1_mods, *test2_mods]

        n_batches = len(barcodes)
        # READ IN THE COUNT MATRICES
        # scRNA-seq of batch
        counts_rnas = []
        for bc in train_batches_barcodes:
            count = scmomat.preprocess(input_train_mod2[bc, hvg_names].layers['counts'].A, modality = "RNA", log = False)
            counts_rnas.append(count)
        for bc in test1_batches_barcodes:
            counts_rnas.append(None)
        for bc in test2_batches_barcodes:
            count = scmomat.preprocess(input_test_mod2[bc, hvg_names].layers['counts'].A, modality = "RNA", log = False)
            counts_rnas.append(count)

        counts_adts = []
        for bc in train_batches_barcodes:
            count = scmomat.preprocess(input_train_mod1[bc,].layers['counts'].A, modality = "ADT", log = True)
            counts_adts.append(count)
        for bc in test1_batches_barcodes:
            count = scmomat.preprocess(input_test_mod1[bc,].layers['counts'].A, modality = "ADT", log = True)
            counts_adts.append(count)
        for bc in test2_batches_barcodes:
            counts_adts.append(None)

        # CREATE THE COUNTS OBJECT
        counts = {"feats_name": feats_name, "nbatches": n_batches, "rna":counts_rnas, "adt": counts_adts}

        #### training
        #------------------------------------------------------------------------------------------------------------------------------------
        # NOTE: Number of latent dimensions, key hyper-parameter, 20~30 works for most of the cases.
        K = 30
        lamb = 0.001 
        T = 4000
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

        labels = []
        for bc in train_batches_barcodes:
            labels.append(df_meta.loc[bc, 'cell_type'].to_numpy())
        for bc in test1_batches_barcodes:
            labels.append(df_meta.loc[bc, 'cell_type'].to_numpy())
            
        for bc in test2_batches_barcodes:
            labels.append(input_test_mod2[bc].obs['cell_type'].to_numpy())

        zs = model.extract_cell_factors()

#         n_neighbors = 100
#         r = None
#         resolution = 0.9
#         knn_indices, knn_dists = scmomat.calc_post_graph(zs, n_neighbors, njobs = 8, r = r)

        ### evaluation
        ad_mosaic = sc.AnnData(np.vstack(zs), obsm={"X_emb":np.vstack(zs)})
        ad_mosaic.obs['batch'] = np.hstack(batches)
        ad_mosaic.obs['mod']   = np.hstack(mods)
        ad_mosaic.obs['mod-batch'] = (ad_mosaic.obs['mod'] + '-' + ad_mosaic.obs.batch).to_numpy()
        ad_mosaic.obs['cell_type'] = np.hstack(labels)

#         ad_mosaic.obsp['connectivities'] = scmomat.utils._compute_connectivities_umap(
#             knn_indices = knn_indices, knn_dists = knn_dists, 
#             n_neighbors = 15, set_op_mix_ratio=1.0, local_connectivity=1.0
#         )
#         ad_mosaic.uns['neighbors'] = {'connectivities_key':'connectivities'}

        # ======================================
        # before harmony
        # ======================================
        print('================================')
        print(f'p={p}, repeat={repeat}')
        print('================================')

        # mosaic eval
        r = eval_mosaic(ad_mosaic, label_key='cell_type', batch_keys=['mod-batch', 'mod'], 
                        use_lisi=True, use_gc=False, use_nmi=False, use_rep='X_emb', use_neighbors=False)
        MOD_LISI1.append(r['mod_LISI'])
        BATCH_LISI1.append(r['mod-batch_LISI'])
        
        # nmi, ari using nmi search
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb', use_neighbors=False,
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))
        NMI1.append(nmi)
        ARI1.append(ari)
        
        ## ================
        #  After harmony
        ## ================
        from preprocessing import harmony
        ad_mosaic_df = pd.DataFrame(ad_mosaic.obsm['X_emb'], index=ad_mosaic.obs_names)
        ad_mosaic_df['batch'] = ad_mosaic.obs['mod-batch'].to_numpy()
        ad_mosaic.obsm['X_emb_harmony'] = harmony([ad_mosaic_df])[0]
        
        r = eval_mosaic(ad_mosaic, label_key='cell_type', batch_keys=['mod-batch', 'mod'], 
            use_lisi=True, use_rep='X_emb_harmony', use_neighbors=False, use_gc=False, use_nmi=False)  # mod-lisi = batch_lisi
        MOD_LISI2.append(r['mod_LISI'])
        BATCH_LISI2.append(r['mod-batch_LISI'])

        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb_harmony', use_neighbors=False,
            use='nmi', nmi_method='arithmetic')
        print("nmi={:.4f}, ari={:.4f}".format(nmi, ari))
        NMI2.append(nmi)
        ARI2.append(ari)
        
        INDS.append(f'p={p}-r={repeat}')
        
df_be = pd.DataFrame({'nmi':NMI1, 'ari':ARI1, 'mod-lisi':MOD_LISI1, 'batch-lisi':BATCH_LISI1}, index=INDS)
df_af = pd.DataFrame({'nmi':NMI2, 'ari':ARI2, 'mod-lisi':MOD_LISI2, 'batch-lisi':BATCH_LISI2}, index=INDS)
df_be.to_csv('/home/yanxh/gitrepo/multi-omics-matching/Visualization/outputs/case4/cite/case4_scmomat.csv', index=True)
df_af.to_csv('/home/yanxh/gitrepo/multi-omics-matching/Visualization/outputs/case4/cite/case4_scmomat_harmony.csv', index=True)
                      

        