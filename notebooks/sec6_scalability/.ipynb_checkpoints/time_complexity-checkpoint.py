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
import h5py
import datetime
from os.path import join

# sys.path.insert(0, '../..')
from ACE.ace import ACE
from ACE.preprocessing import lsiTransformer, ADTransformer, HARMONY
from ACE.evaluation import eval_clustering, eval_lisi, eval_bridge, eval_bridge_above2

print('Reading `mtx` files...')
_path = '../../data/COVID-19/Bridge/cite.h5'
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
    
    adt_norm_data = np.array(f['adt_norm_data'], dtype=np.float32)
    # adt_count_data = np.array(f['adt_count_data'], dtype=np.float32)
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
ad_cite.obsm['adt'] = adt_norm_data

input_dir = '../../data/Other_inputs/sec6'
log_dir = './checkpoints'

for rate in [0.01, 0.1, 0.2, 0.4, 0.8, 1.0]:
    smp_names = pd.read_csv(join(input_dir, f'names_{rate}.csv'))['0'].values
    n_smp = len(smp_names)
    n_interval = n_smp // 3
    ad_cite_subset = ad_cite[smp_names].copy()
    batch = ad_cite_subset.obs.batch.to_numpy()

    gex = sc.AnnData(ad_cite_subset[:n_interval].X.copy(), obs=ad_cite_subset[:n_interval].obs.copy())
    other = sc.AnnData(ad_cite_subset[:n_interval].obsm['adt'].copy(), obs=ad_cite_subset[:n_interval].obs.copy())
    gex_test = sc.AnnData(
        ad_cite_subset[n_interval:int(2*n_interval)].X.copy(), 
        obs=ad_cite_subset[n_interval:int(2*n_interval)].obs.copy()
    )
    other_test = sc.AnnData(
        ad_cite_subset[int(2*n_interval):int(3*n_interval)].obsm['adt'].copy(), 
        obs=ad_cite_subset[int(2*n_interval):int(3*n_interval)].obs.copy()
    )

    print('=======> preprocessing')
    gex_all = sc.concat([gex, gex_test])
    print("gex_all.max = ", gex_all.X.max())

    sc.pp.normalize_total(gex_all, target_sum=1e4)
    sc.pp.log1p(gex_all)
    sc.pp.pca(gex_all, n_comps=50)
    gex_all_lsi_df = pd.DataFrame(gex_all.obsm['X_pca'], index=gex_all.obs_names.to_numpy())
    gex_all.obsm['dimred'] = HARMONY(gex_all_lsi_df, gex_all.obs.batch.to_list(), use_gpu=True)
    gex.obsm['dimred'], gex_test.obsm['dimred'] = gex_all.obsm['dimred'][:gex.shape[0]], gex_all.obsm['dimred'][gex.shape[0]:]

    del gex_all_lsi_df
    gc.collect()

    # preprocessing, protein
    other_all = sc.concat([other, other_test])

    other_all_df = pd.DataFrame(other_all.X, index=other_all.obs_names)
    other_all.obsm['X_pp'] = HARMONY(other_all_df, other_all.obs.batch.to_list(), use_gpu=True)
    other.obsm['X_pp'], other_test.obsm['X_pp'] = other_all.obsm['X_pp'][:other.shape[0]], other_all.obsm['X_pp'][other.shape[0]:]

    del other_all_df
    gc.collect()

    st_time = datetime.datetime.now()
    T = 0.1
    lat = 128
    bs = 512
    
    n_parts = 3
    modBatch_dict = {
        'rna': [gex, gex_test, None],
        'adt': [other, None, other_test]
    }

    useReps_dict = {
        'rna': 'dimred',
        'adt': 'X_pp'
    }

    ##### stage 1
    T = 0.1
    model = ACE(
        modBatch_dict=modBatch_dict,  
        useReps_dict=useReps_dict,  
        batch_key='batch', 
        layers_dims = {'rna': [1024, 512], 'adt':[512, 2048]}, # consistent across all experiments
        dropout_rates = {'rna':[0.2, 0.2], 'adt':[0.2, 0.2]},  # consistent across all experiments
        T=math.log(1./T), T_learnable=False, log_dir=log_dir,  # cover again and again
        n_latent1=lat, n_latent2=lat, seed=1234, num_workers=6
    )

    model.stage1_fit(
        batch_size=bs,
        epochs=80,
        lr=2e-4, 
        log_step=4, 
    )

    # stage1 inference
    ad_integ = model.stage1_infer(
        modBatch_dict, useReps_dict, output_key='stage1_emb', 
        specify_mods_perBatch=[['rna'], ['rna'], ['adt']]
    )

    ##### stage 2
    model.stage2_fit(
        batch_size=1024,
        epochs=10,
        lr=1.75e-4, 
        log_step=2, 
        obvious_be=True,
    )
    
    ed_time = datetime.datetime.now()
    print("==========================")
    print(f'rate={rate}')
    print('Time cost: ', (ed_time - st_time).total_seconds())
    print('==========================')