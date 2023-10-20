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
from os.path import join

# sys.path.insert(0, '../..')
from ACE.ace import ACE
import ACE.utils as utls
from ACE.preprocessing import lsiTransformer, ADTransformer, HARMONY
from ACE.evaluation import eval_clustering, eval_lisi, eval_bridge, eval_bridge_above2

def get_matrix(h5seurat_path, matrix): 
    f = h5py.File(h5seurat_path, 'r')
    matrix = f['assays/SCT/' + matrix]
    data = matrix['data']
    indices = matrix['indices']
    indptr = matrix['indptr']
    mx = sps.csr_matrix((data, indices, indptr))
    mx.indices = mx.indices.astype(np.dtype('int64'))
    mx.indptr = mx.indptr.astype(np.dtype('int64'))
    return mx

def transform_adtNames(ad, vardict):
    ad2 = ad.copy()
    var_names = ad2.var_names.to_numpy()
    for k,v in vardict.items():
        if isinstance(k, tuple):
            ad2[:, k[0]].X = ad2[:, k].X.mean(axis=1)
            var_names[np.where(ad2.var_names == k[0])[0]] = v
        else:
            var_names[np.where(ad2.var_names == k)[0]] = v
    ad2.var_names = var_names
    return ad2

def decode_obs(ad, obs=[]):
    ad.obs_names = [_.decode('utf-8') if isinstance(_, bytes) else _ for _ in ad.obs_names]
    for b in obs:
        ad.obs[b] =  [_.decode('utf-8') if isinstance(_, bytes) else _ for _ in ad.obs[b].to_numpy()]
        
def decode_var(ad):
    ad.var_names = [_.decode('utf-8') if isinstance(_, bytes) else _ for _ in ad.var_names.to_numpy()]

# loading data
# =========================
# loading CITE-seq2 dataset
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
    protein_names = np.array(f['protein_names'], dtype='S32').astype('str')
    
    cite_meta_data = pd.DataFrame(
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
    cite_umap = np.array(f['umap'], dtype=np.float32).T
    
gex = sc.AnnData(rna_count_data, obs=cite_meta_data.loc[cell_names])
gex.var_names = rna_names
other = sc.AnnData(adt_norm_data, obs=cite_meta_data.loc[cell_names])
other.var_names = protein_names
gex.obsm['wnn.umap'] = other.obsm['wnn.umap'] = cite_umap

gex.obs['batch'] = gex.obs.donor + '-' + gex.obs.time
other.obs['batch'] = other.obs.donor + '-' + other.obs.time

del rna_count_data, adt_norm_data
gc.collect()

# =========================
# loading RNA atlas
_path = '../../data/COVID-19/RNA-Atlas/yao_2021_processed.HDF5'
with h5py.File(_path, 'r') as f:
    print(f['reductions'].keys())
#     rnaAtlas_norm_data = get_matrix(_path, 'data')
    rnaAtlas_count_data = get_matrix(_path, 'counts')
    rnaAtlas_rna_names = f['assays/SCT/features'][:]

    cell_names = f['cell.names'][:]
    meta_dict = dict()
    for k in f['meta.data'].keys():
        if k != '_index':
            try:
                v = f[f'meta.data/{k}'][:] 
            except:
                v = f[f'meta.data/{k}/values'][:] 
            meta_dict[k] = v 
      
    meta_data = pd.DataFrame(meta_dict, index=f['meta.data']['_index'][:])
    ref_umapReduc = f['reductions']['ref.umap']['cell.embeddings'][:].T
    umap_Reduc = f['reductions']['rna.umap']['cell.embeddings'][:].T
    rnaAtlas_variable_features = (f['assays/SCT/variable.features'])[:]

gex_test = sc.AnnData(rnaAtlas_count_data, obs=meta_data.loc[cell_names])
gex_test.var_names = rnaAtlas_rna_names
gex_test.obsm['original_umap'] = umap_Reduc
gex_test.obsm['ref_umap'] = ref_umapReduc
gex_test.var['highly_variable'] = np.in1d(rnaAtlas_rna_names, rnaAtlas_variable_features) 

del rnaAtlas_count_data
gc.collect()

# ============================
# loading ADT atlas, after down-sampling
_path = '../../data/CYTOF/sel_data.h5'
with h5py.File(_path, 'r') as f:
    cytof_adt_norm_data = np.array(f['norm_data'], dtype=np.float32)
    cytof_umap = np.array(f['umap'], dtype=np.float32)
    cytof_protein_names = np.array(f['protein_names'], dtype='S32').astype('str')

    adt_test_meta = pd.DataFrame(
        dict(
            cytof_sample_id = np.array(f['sample_id'], dtype=np.int16),
            cytof_condition = np.array(f['condition'], dtype=np.int16),
            cytof_patient_id = np.array(f['patient_id'], dtype=np.int16),
            cytof_batch = np.array(f['batch'], dtype=np.int16),
            cytof_combat_id = np.array(f['COMBAT_ID_Time'], dtype='S32').astype('str'),
            cytof_prior = np.array(f['CyTOF_priority'], dtype=np.int16),
            cytof_major_cell_type = np.array(f['major_cell_type'], dtype='S32').astype('str'),
            cytof_fine_cluster_id = np.array(f['fine_cluster_id'], dtype='S32').astype('str')
        ),
        index=np.array(f['cellID'], dtype=np.int32).astype('str')
    )

other_test = sc.AnnData(cytof_adt_norm_data, obs=adt_test_meta)
other_test.var_names = cytof_protein_names

del cytof_adt_norm_data
gc.collect()

# ============================
# process gene names
gex.var_names = [_.upper() for _ in gex.var_names]
gex_test.var_names = [_.decode('utf-8').upper() for _ in gex_test.var_names]
gex.var_names_make_unique()
gex_test.var_names_make_unique()

shared_gene_names = np.intersect1d(
    gex.var_names, 
    gex_test.var_names
)

gex = gex[:, shared_gene_names].copy()
gex_test = gex_test[:, shared_gene_names].copy()

# processing protein names
other.var_names = [_.upper() for _ in other.var_names]
other_test.var_names = [_.upper() for _ in other_test.var_names]
other.var_names_make_unique()
other_test.var_names_make_unique()

var_dict = {
    ('CD3-1', 'CD3-2'): 'CD3',
    ('CD38-1', 'CD38-2'): 'CD38',
    ('CD4-1', 'CD4-2'): 'CD4',
    ('CD45-1', 'CD45-2'): 'CD45',
    ('CD56-1', 'CD56-2'): 'CD56',
    ('CD66A/C/E', 'CD66B'): 'CD66',
    'HLD-DR': 'HLA_DR',
    'IGD': 'IGD_TCRGD',
    'SIGLEC-8': 'SIGLEC_8',
    'TCR-V-7.2' : 'VA7_2'
}

other = transform_adtNames(other, var_dict)

shared_protein_names = np.intersect1d(
    other.var_names, 
    other_test.var_names
)

other = other[:, shared_protein_names].copy()
other_test = other_test[:, shared_protein_names].copy()
assert other.shape[1] == 31, 'incorrect protein features'

decode_obs(gex_test, list(gex_test.obs.columns))
decode_obs(other_test, list(other_test.obs.columns))
decode_var(gex_test)
decode_var(other_test)

print(f'Gex in {gex.shape}, other in {other.shape}, gex_test in {gex_test.shape}, other_test in {other_test.shape}')
log_dir = './checkpoints/yao'
os.makedirs(log_dir, exist_ok=True)

print('=======> preprocessing')
if not os.path.exists(join(log_dir, 'gex_train_input.npy')):
    # ============================
    # preprocessing, gex
    gex_test.obs['source'] = gex_test.obs['sample'].to_numpy()
    gex.obs['source'] = gex.obs['batch'].to_numpy()
    gex_all = sc.concat([gex, gex_test])
    print("gex_all.max = ", gex_all.X.max())

    sc.pp.normalize_total(gex_all, target_sum=1e4)
    sc.pp.log1p(gex_all)
    sc.pp.highly_variable_genes(gex_all, n_top_genes=5000)
    sc.pp.pca(gex_all, n_comps=50)
    gex_all_lsi_df = pd.DataFrame(gex_all.obsm['X_pca'], index=gex_all.obs_names.to_numpy())
    gex_all.obsm['dimred_be'] = HARMONY(gex_all_lsi_df, gex_all.obs.source.to_list(), use_gpu=True)    
    gex.obsm['dimred_be'], gex_test.obsm['dimred_be'] = gex_all.obsm['dimred_be'][:gex.shape[0]], gex_all.obsm['dimred_be'][gex.shape[0]:]

    del gex_all_lsi_df
    gc.collect()

    # preprocessing, protein
    other_test.obs['source'] = 'test' # other_test_sub.obs['cytof_combat_id'].to_numpy()
    other.obs['source'] = other.obs['batch'].to_numpy()
    other_all = sc.concat([other, other_test])

    other_all_df = pd.DataFrame(other_all.X, index=other_all.obs_names)
    other_all.obsm['X_be'] = HARMONY(other_all_df, other_all.obs['source'].to_list(), use_gpu=True)
    other.obsm['X_be'], other_test.obsm['X_be'] = other_all.obsm['X_be'][:other.shape[0]], other_all.obsm['X_be'][other.shape[0]:]

    del other_all_df
    gc.collect()

    np.save(join(log_dir, 'gex_train_input.npy'), gex.obsm['dimred_be'])
    np.save(join(log_dir, 'gex_test_input.npy'),  gex_test.obsm['dimred_be'])
    np.save(join(log_dir, 'other_train_input.npy'), other.obsm['X_be'])
    np.save(join(log_dir, 'other_test_input.npy'), other_test.obsm['X_be'])
else:
    gex.obsm['dimred_be']        = np.load(join(log_dir, 'gex_train_input.npy'))
    gex_test.obsm['dimred_be']   = np.load(join(log_dir, 'gex_test_input.npy'))
    other.obsm['X_be']      = np.load(join(log_dir, 'other_train_input.npy'))
    other_test.obsm['X_be'] = np.load(join(log_dir, 'other_test_input.npy'))
    
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
    T=math.log(1./T), T_learnable=False, log_dir=log_dir,
    n_latent1=128, n_latent2=128, seed=1234, num_workers=6
)

# model.stage1_fit(
#     batch_size=1024,
#     epochs=8,
#     lr=2e-4, 
#     log_step=4, 
# )

# loading pretrained weights
utls.load_model(model.stage1_encoders.encoders, model.mod_names, log_dir, ckpt_id=8, stage_label='stage1')

# stage1 inference
ad_integ = model.stage1_infer(
    modBatch_dict, useReps_dict, output_key='stage1_emb', 
    specify_mods_perBatch=[['rna'], ['rna'], ['adt']]
)

##### stage 2
# model.stage2_fit(
#     batch_size=1024,
#     epochs=4,
#     lr=1.75e-4, 
#     log_step=2, 
#     obvious_be=True,
# )

# loading pretrained weights
utls.load_model(model.stage2_encoders, model.mod_names, log_dir, ckpt_id=4, stage_label='stage2')

ad_integ2 = model.stage2_infer(
    modBatch_dict, useReps_dict, output_key1='stage1_emb', output_key2='stage2_emb', 
    knn=2, mod_weights={'rna':0.5, 'adt':0.5}
)

ad_integ2.write_h5ad(join(log_dir, 'adata_integ2.h5ad'))