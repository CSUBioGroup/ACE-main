from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os
import gc
import math
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scanpy as sc
import scipy.io as sio

from os.path import join
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/ACE/reproduce/evaluation')
from evaluation import eval_mosaic, eval_specific_mod, eval_bridge, print_results, eval_asw, eval_lisi, eval_clustering

def UniInt_list(xs):
    inter = set(xs[0])
    union = set(xs[0])

    for x in xs[1:]:
        inter = inter & set(x)
        union = union | set(x)
    return inter, union

def generate(set1, set2, set3, p=0.1):
#     set1, set2, set3 = ad1.obs.cell_type.unique(), ad2.obs.cell_type.unique(), ad3.obs.cell_type.unique()
    all_inter_set = list(set(set1) & set(set2) & set(set3))
    all_set = list(set(set1) | set(set2) | set(set3))
    
    n_inter = min(math.floor(p*len(all_set)), len(all_inter_set))
    # n_inter = np.random.randint(2, max_n_inter+1)
    n_union = min(len(all_set), math.ceil(n_inter/p))
    inter_set = list(np.random.choice(all_inter_set, n_inter, replace=False))

    sets = [inter_set, inter_set, inter_set]  # add intersection
    leave_set = np.random.choice(list(set(all_set) - set(inter_set)), n_union-n_inter, replace=False)
    for t in leave_set:
        in_mask = np.array([(t in x) for x in [set1, set2, set3]])
        max_n = min(in_mask.sum()+1, 3)
        n_insert = np.random.randint(1, max_n) # 1 or 2
        insert_ixs = np.random.choice(np.where(in_mask)[0], n_insert, replace=False)
        sets = [_+[t] if i in insert_ixs else _ for i,_ in enumerate(sets)]
    return sets

# Path to the data directory
dat_dir = "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/"
data_dir = os.path.join(dat_dir, "match_modality/openproblems_bmmc_cite_phase2_mod2")
DIR = '/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/cite'
os.makedirs(DIR, exist_ok=True)

# reading data
print('Reading `h5ad` files...')
input_train_mod1_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod1.h5ad'))
input_train_mod2_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod2.h5ad'))
input_train_sol = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_sol.h5ad'))
input_test_mod1_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_mod1.h5ad'))
input_test_mod2_0 = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_mod2.h5ad'))

print("Unscrambling training cells...")
ord = input_train_sol.X.tocsr().indices
train_mod2_ord = ord.copy()
if "pairing_ix" in input_train_sol.uns:
    assert np.all(ord == np.argsort(input_train_sol.uns["pairing_ix"]))
input_train_mod2_0 = input_train_mod2_0[ord, :].copy()
input_train_mod2_0.obs_names = input_train_mod1_0.obs_names

meta_dir = '/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main'
df_meta = pd.read_csv(os.path.join(meta_dir, 'output/datasets/cite_meta.csv'), index_col=0)
input_train_mod1_0.obs[['batch', 'cell_type']] = df_meta.loc[input_train_mod1_0.obs_names, ['batch', 'cell_type']].to_numpy()
input_train_mod2_0.obs[['batch', 'cell_type']] = df_meta.loc[input_train_mod2_0.obs_names, ['batch', 'cell_type']].to_numpy()

input_test_sol = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_sol.h5ad'))
test_ord = input_test_sol.X.tocsr().indices
assert (test_ord == np.argsort(input_test_sol.uns['pairing_ix'])).all()
input_test_mod1_0.obs[['batch', 'cell_type']] = df_meta.loc[input_test_mod1_0.obs_names, ['batch', 'cell_type']].to_numpy()
input_test_mod2_0.obs['cell_type'] = input_test_mod1_0.obs.cell_type[np.argsort(test_ord)].to_numpy()
assert np.all(input_test_mod1_0.obs.cell_type.to_numpy() == input_test_mod2_0[test_ord].obs.cell_type.to_numpy())

######### reducing bridge number here
for p in [0.1, 0.2, 0.4, 0.8]:  
    for repeat in range(3):
        set1 = input_train_mod1_0.obs.cell_type.unique()  # multiome
        set2 = input_test_mod2_0.obs.cell_type.unique()   # rna
        set3 = input_test_mod1_0.obs.cell_type.unique()   # adt
        sets = generate(set1, set2, set3, p=p)

        train_idx = np.where(input_train_mod1_0.obs.cell_type.isin(sets[0]))[0]
        test_rna_idx = np.where(input_test_mod2_0.obs.cell_type.isin(sets[1]))[0]
        test_adt_idx = np.where(input_test_mod1_0.obs.cell_type.isin(sets[2]))[0]
        print(f'training size={train_idx.size}, test_rna size={test_rna_idx.size}, test_adt size={test_adt_idx.size}')
        s1, s2 = UniInt_list(sets)
        print('intersected type ratio: ', len(s1) / len(s2))

        input_train_mod1 = input_train_mod1_0[train_idx, ].copy()
        input_train_mod2 = input_train_mod2_0[train_idx, ].copy()
        input_test_mod1  = input_test_mod1_0[test_adt_idx, ].copy()
        input_test_mod2  = input_test_mod2_0[test_rna_idx, ].copy()

        # hvgs
        sc.pp.highly_variable_genes(input_train_mod2, layer='counts', flavor='seurat_v3', n_top_genes=5000, batch_key='batch')
        hvg_names = input_train_mod2.var.query('highly_variable').index.to_numpy()

        mult_rna_count = input_train_mod2[:, hvg_names].layers["counts"].astype(np.float32)
        mult_adt_count = input_train_mod1.layers["counts"].astype(np.float32)
        mult_rna_count = sps.csr_matrix(mult_rna_count)   # this fucking thing must be in csr format, otherwise, training will be fucking very 
        mult_adt_count = sps.csr_matrix(mult_adt_count)   # slowly, fuck, fuck, fuck
        mult_barcode = input_train_mod2.obs_names.to_numpy()

        rna_feature = hvg_names
        adt_feature = input_train_mod1.var_names.to_numpy()
        mult_rna = SingleData("GeneExpr", "Multiome", rna_feature, mult_rna_count, mult_barcode)
        mult_adt = SingleData("ADT", "Multiome", adt_feature, mult_adt_count, mult_barcode)

        single_rna_count = input_test_mod2[:, hvg_names].layers["counts"].astype(np.float32)
        single_adt_count = input_test_mod1.layers["counts"].astype(np.float32)
        single_rna_count = sps.csr_matrix(single_rna_count)
        single_adt_count = sps.csr_matrix(single_adt_count)
        single_rna_barcode = input_test_mod2.obs_names.to_numpy()
        single_adt_barcode = input_test_mod1.obs_names.to_numpy()

        single_rna = SingleData("GeneExpr", "Single-GEX", rna_feature, single_rna_count, single_rna_barcode)
        single_adt = SingleData("ADT", "Single-ADT", adt_feature, single_adt_count, single_adt_barcode)
        multi_dt = MultiomicDataset.from_singledata(
            single_rna, single_adt, mult_adt, mult_rna)

        # saving for reproducibility
        np.save(join(DIR, f'p={p}_r={repeat}_new_train_idx.npy'), train_idx)
        np.save(join(DIR, f'p={p}_r={repeat}_test_rna_idx.npy'), test_rna_idx)
        np.save(join(DIR, f'p={p}_r={repeat}_test_adt_idx.npy'), test_adt_idx)

        model = Cobolt(dataset=multi_dt, lr=0.0001, n_latent=10, batch_size=128) # all default
        model.train(num_epochs=100)
        model.calc_all_latent()
        latent = model.get_all_latent()

        latent_barcode = np.array([_.split('~')[1] for _ in latent[1]])
        df_latent = pd.DataFrame(latent[0], index=latent_barcode)

        gex_test = sc.AnnData(single_rna_count, obs=input_test_mod2.obs.copy())
        gex_test.obs_names = single_rna_barcode
        other_test = sc.AnnData(single_adt_count, obs=input_test_mod1.obs.copy())
        other_test.obs_names = single_adt_barcode

        ad_mult = sc.AnnData(mult_rna_count, obs=input_train_mod2.obs.copy())
        ad_mult.obs_names = mult_barcode
        ad_mult.obsm['X_emb'] = df_latent.loc[ad_mult.obs_names, :].values
        gex_test.obsm['X_emb'] = df_latent.loc[gex_test.obs_names, :].values
        other_test.obsm['X_emb'] = df_latent.loc[other_test.obs_names, :].values

        ad_mult.obs['mod'] = 'multiome'
        gex_test.obs['mod'] = 'gex'
        other_test.obs['mod'] = 'other'
        ad_mult.obs['mod-batch'] = ad_mult.obs.batch.apply(lambda x: 'multiome'+'-'+x).to_numpy()
        gex_test.obs['mod-batch'] = gex_test.obs.batch.apply(lambda x: 'gex'+'-'+x).to_numpy()
        other_test.obs['mod-batch'] = other_test.obs.batch.apply(lambda x: 'other'+'-'+x).to_numpy()

        # ======================================
        # evaluation
        # ======================================
        print('================================')
        print(f'p={p}, repeat={repeat}')
        print('================================')

        # without harmony
        print('===============Before harmony==============')
        ad_mosaic = sc.concat([ad_mult, gex_test, other_test])
        r = eval_mosaic(ad_mosaic, label_key='cell_type', lisi_keys=['mod-batch', 'mod'], 
            use_rep='X_emb', use_gc=False, use_nmi=False)

        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))

        # with harmony
        print('===============After harmony==============')
        from preprocessing import harmony
        ad_mosaic = sc.concat([ad_mult, gex_test, other_test])

        ad_mosaic_df = pd.DataFrame(ad_mosaic.obsm['X_emb'], index=ad_mosaic.obs_names)
        ad_mosaic_df['batch'] = ad_mosaic.obs['mod-batch'].to_numpy()
        ad_mosaic.obsm['X_emb_harmony'] = harmony([ad_mosaic_df])[0]

        r = eval_mosaic(ad_mosaic, label_key='cell_type', 
            lisi_keys=['mod-batch', 'mod'], use_rep='X_emb_harmony', use_nmi=False, use_gc=False)
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb_harmony',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))