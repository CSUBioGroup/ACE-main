import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import scipy.sparse as sps
import scipy.io as sio
import os
import sys
import math
import gc
from os.path import join

# sys.path.insert(0, '../..')
from ACE.ace import ACE
from ACE.preprocessing import lsiTransformer, ADTransformer, HARMONY
from ACE.evaluation import eval_clustering, eval_lisi, eval_bridge, eval_bridge_above2

# Path to the data directory
# root_dir = '/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main'
data_dir = "../../data/DOGMA"

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
meta_data.columns = ['batch', 'cell_type', 'cell_type.l2']

# train_idx = np.where((meta_data.batch.str.lower()=='control').to_numpy())[0]
test_idx  = np.where((meta_data.batch.str.lower()=='stim').to_numpy())[0]

log_dir = '../../outputs/trimodal_bridge_number_exp'

for del_size in [0.1, 0.2, 0.4, 0.8]:
    for repeat in range(3):
        new_train_idx = np.load(f'../../data/Other_inputs/trimodal_bridge_num_exp/DelSize={del_size}_r={repeat}_ids.npy')
        save_dir = join(log_dir, f'DelSize={del_size}_r={repeat}')
        os.makedirs(save_dir, exist_ok=True)

        ad_mult_rna = sc.AnnData(rna_count_mat[new_train_idx], obs=meta_data.iloc[new_train_idx])
        ad_rna_test = sc.AnnData(rna_count_mat[test_idx], obs=meta_data.iloc[test_idx])
        ad_mult_adt = sc.AnnData(adt_count_mat[new_train_idx], obs=meta_data.iloc[new_train_idx])
        ad_adt_test = sc.AnnData(adt_count_mat[test_idx], obs=meta_data.iloc[test_idx])
        ad_mult_atac = sc.AnnData(atac_count_mat[new_train_idx], obs=meta_data.iloc[new_train_idx])
        ad_atac_test = sc.AnnData(atac_count_mat[test_idx], obs=meta_data.iloc[test_idx])

        if not os.path.exists(join(save_dir, 'gex_train_input.npy')):
            ### GEX preprocessing
            gex_all = sc.concat([ad_mult_rna, ad_rna_test])
            lsi_former = lsiTransformer(192, drop_first=True, use_highly_variable=False, log=True, norm=True, 
                                        z_score=True, tfidf=False, svd=True, use_counts=False
            )
            gex_all_lsi_df = lsi_former.fit_transform(gex_all)
            gex_all_dimred = HARMONY(gex_all_lsi_df, gex_all.obs.batch.to_list(), use_gpu=True)
            ad_mult_rna.obsm['dimred_be'], ad_rna_test.obsm['dimred_be'] = gex_all_dimred[:ad_mult_rna.shape[0]], gex_all_dimred[ad_mult_rna.shape[0]:]

            del gex_all, gex_all_lsi_df, gex_all_dimred
            gc.collect()

            ### ADT preprocessing
            adt_all = sc.concat([ad_mult_adt, ad_adt_test])
            adt_lsi_former = ADTransformer(adt_all.shape[1], drop_first=False, svd=False, log=True, 
                                             norm=True, z_score=True, use_counts=False)
            adt_all_df = adt_lsi_former.fit_transform(adt_all)
            adt_all_arr = HARMONY(adt_all_df, adt_all.obs.batch.to_list(), use_gpu=True)
            ad_mult_adt.obsm['X_be'], ad_adt_test.obsm['X_be'] = adt_all_arr[:ad_mult_adt.shape[0]], adt_all_arr[ad_mult_adt.shape[0]:]
            
            del adt_all, adt_all_df, adt_all_arr
            gc.collect()

            ### ATAC preprocessing
            atac_all = sc.concat([ad_mult_atac, ad_atac_test])
            atac_lsi_former = lsiTransformer(192, drop_first=True, use_highly_variable=False, log=True, norm=True, 
                                        z_score=True, tfidf=True, svd=True, use_counts=False
            )
            atac_all_df = atac_lsi_former.fit_transform(atac_all)
            atac_all_arr = HARMONY(atac_all_df, atac_all.obs.batch.to_list(), use_gpu=True)
            ad_mult_atac.obsm['dimred_be'], ad_atac_test.obsm['dimred_be'] = atac_all_arr[:ad_mult_atac.shape[0]], atac_all_arr[ad_mult_atac.shape[0]:]

            del atac_all, atac_all_df, atac_all_arr
            gc.collect()

            np.save(join(save_dir, 'gex_train_input.npy'), ad_mult_rna.obsm['dimred_be'])
            np.save(join(save_dir, 'gex_test_input.npy'),  ad_rna_test.obsm['dimred_be'])
            np.save(join(save_dir, 'adt_train_input.npy'), ad_mult_adt.obsm['X_be'])
            np.save(join(save_dir, 'adt_test_input.npy'), ad_adt_test.obsm['X_be'])
            np.save(join(save_dir, 'atac_train_input.npy'), ad_mult_atac.obsm['dimred_be'])
            np.save(join(save_dir, 'atac_test_input.npy'), ad_atac_test.obsm['dimred_be'])
        else:
            ad_mult_rna.obsm['dimred_be'] = np.load(join(save_dir, 'gex_train_input.npy'))
            ad_rna_test.obsm['dimred_be'] = np.load(join(save_dir, 'gex_test_input.npy'))
            ad_mult_adt.obsm['X_be'] = np.load(join(save_dir, 'adt_train_input.npy'))
            ad_adt_test.obsm['X_be'] = np.load(join(save_dir, 'adt_test_input.npy'))
            ad_mult_atac.obsm['dimred_be']= np.load(join(save_dir, 'atac_train_input.npy'))
            ad_atac_test.obsm['dimred_be']= np.load(join(save_dir, 'atac_test_input.npy'))
        
        n_parts = 4
        modBatch_dict = {
            'rna': [ad_mult_rna, ad_rna_test, None, None],
            'adt': [ad_mult_adt, None, ad_adt_test, None],
            'atac':[ad_mult_atac, None, None, ad_atac_test]
        }

        useReps_dict = {
            'rna': 'dimred_be',
            'adt': 'X_be',
            'atac': 'dimred_be'
        }
        
        ##### stage 1
        T = 0.1
        model = ACE(
            modBatch_dict=modBatch_dict,  
            useReps_dict=useReps_dict,  
            batch_key='batch', 
            layers_dims = {'rna': [1024, 512], 'adt':[512, 2048], 'atac':[1024, 512]}, # consistent across all experiments
            dropout_rates = {'rna':[0.2, 0.2], 'adt':[0.2, 0.2],  'atac':[0.2, 0.2]},  # consistent across all experiments
            T=math.log(1./T), T_learnable=False, log_dir=save_dir,
            n_latent1=40, n_latent2=256, seed=1234, 
            num_workers=6
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
            specify_mods_perBatch=[['rna'], ['rna'], ['adt'], ['atac']]
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
            knn=2, mod_weights={'rna':1/3, 'adt':1/3, 'atac':1/3}
        )
        
        ##### evaluation
        ### nmi, ari
        ad_integ.obsm['stage1_emb'] = ad_integ.X.copy()
        ad_integ2.obsm['final_emb'] = ad_integ2.X.copy()

        nmi1, ari1 = eval_clustering(
            ad_integ, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='stage1_emb',
            use='nmi', nmi_method='arithmetic')

        nmi2, ari2 = eval_clustering(
            ad_integ2, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='final_emb',
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
        
        ### mod alignment
        modBatch_dict['rna'][1].uns['domain'] = 'rna'
        modBatch_dict['adt'][2].uns['domain'] = 'adt'
        modBatch_dict['atac'][3].uns['domain'] = 'atac'

        bridge_res = eval_bridge_above2(
                [modBatch_dict['rna'][1], modBatch_dict['adt'][2], modBatch_dict['atac'][3]],
                label_key='cell_type',
                batch_key='batch',
                mod_key='domain',
                use_rep='stage1_emb',
                use_fosc=True, use_acc=False, use_score=True,
            )