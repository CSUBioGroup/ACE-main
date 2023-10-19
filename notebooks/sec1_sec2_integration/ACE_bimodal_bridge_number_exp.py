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

sys.path.insert(0, '../..')
from src.ace import ACE
from src.preprocessing import lsiTransformer, ADTransformer, HARMONY
from src.evaluation import eval_clustering, eval_lisi, eval_bridge, eval_bridge_above2

root_dir = '../../data/CITE'
other_dir = '../../data/Other_inputs'
dataset_path = os.path.join(root_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_')

par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'input_train_sol': f'{dataset_path}train_sol.h5ad',
    'input_test_sol': f'{dataset_path}test_sol.h5ad',
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
}

print('Reading `h5ad` files...')
input_train_mod1_0 = sc.read_h5ad(par['input_train_mod1'])
input_train_mod2_0 = sc.read_h5ad(par['input_train_mod2'])
input_train_sol   = sc.read_h5ad(par['input_train_sol'])
input_test_mod1_0 = sc.read_h5ad(par['input_test_mod1'])
input_test_mod2_0 = sc.read_h5ad(par['input_test_mod2'])

print("Unscrambling training cells...")
ord = input_train_sol.X.tocsr().indices
if "pairing_ix" in input_train_sol.uns:
    assert np.all(ord == np.argsort(input_train_sol.uns["pairing_ix"]))
input_train_mod2_0 = input_train_mod2_0[ord, :].copy()
input_train_mod2_0.obs_names = input_train_mod1_0.obs_names

input_test_sol = sc.read_h5ad(par['input_test_sol'])
ord = input_test_sol.X.tocsr().indices
if "pairing_ix" in input_test_sol.uns:
    assert np.all(ord == np.argsort(input_test_sol.uns["pairing_ix"]))
input_test_mod2_0 = input_test_mod2_0[ord, :].copy()
input_test_mod2_0.obs_names = input_test_mod1_0.obs_names    

df_meta = pd.read_csv(os.path.join(root_dir, 'cite_meta.csv'), index_col=0)
log_dir = '../../outputs/bimodal_bridge_num'

for del_size in [0.1, 0.2, 0.4, 0.8]:
    for repeat in range(3):
        sel_cname = np.load(f'{other_dir}/bimodal_bridge_num_exp/cite_DelSize={del_size}_r={repeat}_ids.npy', allow_pickle=True)
        save_dir = join(log_dir, f'DelSize={del_size}_r={repeat}')
        os.makedirs(save_dir, exist_ok=True)

        gex = input_train_mod2_0[sel_cname].copy()
        other = input_train_mod1_0[sel_cname].copy()
        gex_test = input_test_mod2_0.copy()
        other_test = input_test_mod1_0.copy()

        gex.obs = df_meta.loc[gex.obs_names].copy()
        gex_test.obs = df_meta.loc[gex_test.obs_names].copy()
        other.obs = df_meta.loc[other.obs_names].copy()
        other_test.obs = df_meta.loc[other_test.obs_names].copy()

        if not os.path.exists(join(save_dir, 'gex_train_input.npy')):
            ### GEX preprocessing
            gex_all = sc.concat([gex, gex_test])
            lsi_former = lsiTransformer(192, drop_first=True, use_highly_variable=False, log=False, norm=False, 
                                        z_score=True, tfidf=False, svd=True, use_counts=False
            )
            gex_all_dimred_df = lsi_former.fit_transform(gex_all)
            gex_all_dimred = HARMONY(gex_all_dimred_df, gex_all.obs.batch.to_list(), use_gpu=True)
            gex.obsm['dimred_be'], gex_test.obsm['dimred_be'] = gex_all_dimred[:gex.shape[0]], gex_all_dimred[gex.shape[0]:]

            del gex_all, gex_all_dimred, gex_all_dimred_df
            gc.collect()

            ### ADT preprocessing
            other_all = sc.concat([other, other_test])
            other_lsi_former = ADTransformer(other.shape[1], drop_first=False, svd=False, log=False, 
                                             norm=False, z_score=True, use_counts=False)
            other_all_df = other_lsi_former.fit_transform(other_all)
            other_all_arr = HARMONY(other_all_df, other_all.obs.batch.to_list(), use_gpu=True)
            other.obsm['X_be'], other_test.obsm['X_be'] = other_all_arr[:other.shape[0]], other_all_arr[other.shape[0]:]

            del other_all, other_all_df, other_all_arr
            gc.collect()

            np.save(join(save_dir, 'gex_train_input.npy'), gex.obsm['dimred_be'])
            np.save(join(save_dir, 'gex_test_input.npy'),  gex_test.obsm['dimred_be'])
            np.save(join(save_dir, 'other_train_input.npy'), other.obsm['X_be'])
            np.save(join(save_dir, 'other_test_input.npy'), other_test.obsm['X_be'])
        else:
            gex.obsm['dimred_be']        = np.load(join(save_dir, 'gex_train_input.npy'))
            gex_test.obsm['dimred_be']   = np.load(join(save_dir, 'gex_test_input.npy'))
            other.obsm['X_be']      = np.load(join(save_dir, 'other_train_input.npy'))
            other_test.obsm['X_be'] = np.load(join(save_dir, 'other_test_input.npy'))
        
        n_parts = 3
        modBatch_dict = {
            'rna': [gex, gex_test, None],
            'adt': [other, None, other_test]
        }

        useReps_dict = {
            'rna': 'dimred_be',
            'adt': 'X_be'
        }
        
        ##### stage 1
        T = 0.1
        model = ACE(
            modBatch_dict=modBatch_dict,  
            useReps_dict=useReps_dict,  
            batch_key='batch', 
            layers_dims = {'rna': [1024, 512], 'adt':[512, 2048]}, # consistent across all experiments
            dropout_rates = {'rna':[0.2, 0.2], 'adt':[0.2, 0.2]},  # consistent across all experiments
            T=math.log(1./T), T_learnable=False, log_dir=save_dir,
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
                batch_keys=['domain', 'batch'],
                use_rep='stage1_emb', use_neighbors=False,
            )

        df_lisi2 = eval_lisi(
                ad_integ2,
                batch_keys=['domain', 'batch'],
                use_rep='final_emb', use_neighbors=False,
            )

        print('stage 1: domain-lisi={:.4f}, batch-lisi={:.4f}'.format(df_lisi1.domain_LISI[0], df_lisi1.batch_LISI[0]))
        print('stage 2: domain-lisi={:.4f}, batch-lisi={:.4f}'.format(df_lisi2.domain_LISI[0], df_lisi2.batch_LISI[0]))
        
        ### mod alignment
        bridge_dict = eval_bridge(
            modBatch_dict['rna'][1], modBatch_dict['adt'][2],
            label_key='cell_type',
            batch_key='batch',
            use_rep='stage1_emb',
            use_fosc=True, use_acc=False, use_score=True,
        )