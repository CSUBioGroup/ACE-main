import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import scipy.sparse as sps
import scipy.io as sio
import os
import sys
import math
from os.path import join

# sys.path.insert(0, '../..')
from ACE.ace import ACE
from ACE.evaluation import eval_clustering, eval_lisi, eval_bridge, eval_bridge_above2

# Path to the data directory
# root_dir = '../../BM-CITE'
data_dir = '../../BM-CITE'

print('Reading `mtx` files...')
rna_norm_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'rna_mat_norm.mtx')).T)
adt_norm_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'adt_mat_norm.mtx')).T)

rna_names = pd.read_csv(join(data_dir, 'gene_names.csv'))['x'].to_numpy()
adt_names = pd.read_csv(join(data_dir, 'adt_names.csv'))['x'].to_numpy()

cell_names = pd.read_csv(join(data_dir, 'cell_names.csv'))['x'].to_numpy()
meta_data = pd.read_csv(join(data_dir, 'metadata.csv'), index_col=0)
meta_data['batch'] = meta_data['donor'].to_numpy()

# train_idx = np.where((meta_data.batch=='batch1').to_numpy())[0]
# test_idx  = np.where((meta_data.batch=='batch2').to_numpy())[0]

input_dir = '../../data/Other_inputs/bimodal_case3'
log_dir   = '../../outputs/bimodal_case3'
os.makedirs(log_dir, exist_ok=True)

for p in [0.1, 0.2, 0.4, 0.8]:
    for r in range(3):
        new_train_idx = np.load(join(input_dir, f'p={p}_r={r}_new_train_idx.npy'))
        test_rna_idx  = np.load(join(input_dir, f'p={p}_r={r}_test_rna_idx.npy'))
        test_adt_idx  = np.load(join(input_dir, f'p={p}_r={r}_test_adt_idx.npy'))
        save_dir = join(log_dir, f'p={p}_repeat={r}')
        os.makedirs(save_dir, exist_ok=True)
        
        print('Reading `h5ad` files...')
        gex = sc.AnnData(sps.csr_matrix(rna_norm_mat[new_train_idx]), obs=meta_data.iloc[new_train_idx])
        other = sc.AnnData(sps.csr_matrix(adt_norm_mat[new_train_idx]), obs=meta_data.iloc[new_train_idx])
        gex_test = sc.AnnData(sps.csr_matrix(rna_norm_mat[test_rna_idx]), obs=meta_data.iloc[test_rna_idx])
        other_test = sc.AnnData(sps.csr_matrix(adt_norm_mat[test_adt_idx]), obs=meta_data.iloc[test_adt_idx])

        # set var names
        gex.var_names = gex_test.var_names = rna_names
        other.var_names = other_test.var_names = adt_names

        gex.obs = meta_data.loc[gex.obs_names].copy()
        gex_test.obs = meta_data.loc[gex_test.obs_names].copy()
        other.obs = meta_data.loc[other.obs_names].copy()
        other_test.obs = meta_data.loc[other_test.obs_names].copy()
        
        if not os.path.exists(join(save_dir, 'gex_train_input.npy')):
            ### GEX preprocessing
            gex_all = sc.concat([gex, gex_test])
            sc.pp.scale(gex_all)
            sc.pp.pca(gex_all, n_comps=50)
            gex.obsm['X_pca'] = gex_all.obsm['X_pca'][:gex.shape[0]]
            gex_test.obsm['X_pca'] = gex_all.obsm['X_pca'][gex.shape[0]:]

            np.save(join(save_dir, 'gex_train_input.npy'), gex.obsm['X_pca'])
            np.save(join(save_dir, 'gex_test_input.npy'),  gex_test.obsm['X_pca'])
        else:
            gex.obsm['X_pca']        = np.load(join(save_dir, 'gex_train_input.npy'))
            gex_test.obsm['X_pca']   = np.load(join(save_dir, 'gex_test_input.npy'))
            
        n_parts = 3
        modBatch_dict = {
            'rna': [gex, gex_test, None],
            'adt': [other, None, other_test]
        }

        useReps_dict = {
            'rna': 'X_pca',
            'adt': 'X'
        }
        
        ##### stage 1 
        T = 0.1
        model = ACE(
            modBatch_dict=modBatch_dict,  
            useReps_dict=useReps_dict,  
            batch_key='batch', 
            layers_dims = {'rna': [1024, 512], 'adt':[512, 2048]}, # consistent across all experiments
            dropout_rates = {'rna':[0.2, 0.2], 'adt':[0.2, 0.2]},  # consistent across all experiments
            T=math.log(1./T), T_learnable=False, log_dir=log_dir,
            n_latent1=40, n_latent2=256, seed=1234, num_workers=6
        )

        model.stage1_fit(
            batch_size=512,
            epochs=100,
            lr=2e-4, 
            log_step=10, 
        )
        
        # stage1 inference
        ad_integ = model.stage1_infer(
            modBatch_dict, useReps_dict, output_key='stage1_emb', 
            specify_mods_perBatch=[['rna'], ['rna'], ['adt']]
        )
        
        ##### stage 2
        model.stage2_fit(
            batch_size=512,
            epochs=10,
            lr=1.75e-4, 
            log_step=5, 
            obvious_be=True,
        )
        
        ad_integ2 = model.stage2_infer(
            modBatch_dict, useReps_dict, output_key1='stage1_emb', output_key2='stage2_emb', 
            knn=2, mod_weights={'rna':0.5, 'adt':0.5}
        )
        
        ##### evaluation
        ### nmi, ari
        ad_integ.obsm['stage1_emb'] = ad_integ.X.copy()
        ad_integ2.obsm['final_emb'] = ad_integ2.X.copy()

        nmi1, ari1 = eval_clustering(
            ad_integ, label_key='celltype.l2', cluster_key='cluster', resolutions=None, use_rep='stage1_emb',
            use='nmi', nmi_method='arithmetic')

        nmi2, ari2 = eval_clustering(
            ad_integ2, label_key='celltype.l2', cluster_key='cluster', resolutions=None, use_rep='final_emb',
            use='nmi', nmi_method='arithmetic')

        print('stage 1: nmi={:.4f}, ari={:.4f}'.format(nmi1, ari1))
        print('stage 2: nmi={:.4f}, ari={:.4f}'.format(nmi2, ari2))
        
        ### lisi
        df_lisi1 = eval_lisi(
                ad_integ,
                batch_keys=['domain'],
                use_rep='stage1_emb', use_neighbors=False,
            )

        df_lisi2 = eval_lisi(
                ad_integ2,
                batch_keys=['domain'],
                use_rep='final_emb', use_neighbors=False,
            )

        print('stage 1: domain(batch)-lisi={:.4f}'.format(df_lisi1.domain_LISI[0]))
        print('stage 2: domain(batch)-lisi={:.4f}'.format(df_lisi2.domain_LISI[0]))
        
        

        

        
        
