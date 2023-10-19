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

import sys
sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/ACE/reproduce/evaluation')
from evaluation import eval_mosaic, eval_specific_mod, eval_bridge, print_results, eval_asw, eval_lisi, eval_clustering
from preprocessing import harmony

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
data_dir = "/home/yanxh/data/Seurat_demo_data/bm_cite"
save_dir = '/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case3/bm-cite'

# loading data
print('Reading `mtx` files...')
rna_count_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'rna_mat_count.mtx')).T)
adt_count_mat = sps.csr_matrix(sio.mmread(join(data_dir, 'adt_mat_count.mtx')).T)

rna_names = pd.read_csv(join(data_dir, 'gene_names.csv'))['x'].to_numpy()
adt_names = pd.read_csv(join(data_dir, 'adt_names.csv'))['x'].to_numpy()

cell_names = pd.read_csv(join(data_dir, 'cell_names.csv'))['x'].to_numpy()
meta_data = pd.read_csv(join(data_dir, 'metadata.csv'), index_col=0)
meta_data['batch'] = meta_data.donor.to_numpy()

train_idx = np.where((meta_data.batch=='batch1').to_numpy())[0]
test_idx  = np.where((meta_data.batch=='batch2').to_numpy())[0]

for p in [0.1, 0.2, 0.4, 0.8]:
    for repeat in range(3):
        # generate random split of cell types
        set1 = meta_data.loc[cell_names[train_idx], 'celltype.l2'].unique()
        set2 = meta_data.loc[cell_names[test_idx], 'celltype.l2'].unique()
        set3 = meta_data.loc[cell_names[test_idx], 'celltype.l2'].unique()

        sets = generate(set1, set2, set3, p=p)
        new_train_idx = train_idx[meta_data.loc[cell_names[train_idx], 'celltype.l2'].isin(sets[0])]
        test_rna_idx = test_idx[meta_data.loc[cell_names[test_idx], 'celltype.l2'].isin(sets[1])]
        test_adt_idx = test_idx[meta_data.loc[cell_names[test_idx], 'celltype.l2'].isin(sets[2])]
        print(f'training size={new_train_idx.size}, test_rna size={test_rna_idx.size}, test_adt size={test_adt_idx.size}')

        s1, s2 = UniInt_list(sets)
        print('intersected type ratio: ', len(s1) / len(s2))

        mult_rna_count = rna_count_mat[new_train_idx].astype(np.float32)
        mult_adt_count = adt_count_mat[new_train_idx].astype(np.float32)
        mult_rna_count = sps.csr_matrix(mult_rna_count)   # this fucking thing must be in csr format, otherwise, training will be fucking very 
        mult_adt_count = sps.csr_matrix(mult_adt_count)   # slowly, fuck, fuck, fuck
        mult_barcode = cell_names[new_train_idx]

        rna_feature = rna_names
        adt_feature = adt_names

        mult_rna = SingleData("GeneExpr", "Multiome", rna_feature, mult_rna_count, mult_barcode)
        mult_adt = SingleData("ADT", "Multiome", adt_feature, mult_adt_count, mult_barcode)

        single_rna_count = rna_count_mat[test_rna_idx].astype(np.float32)
        single_adt_count = adt_count_mat[test_adt_idx].astype(np.float32)
        single_rna_count = sps.csr_matrix(single_rna_count)
        single_adt_count = sps.csr_matrix(single_adt_count)
        single_rna_barcode = [_+'_rna' for _ in cell_names[test_rna_idx]]
        single_adt_barcode = [_+'_adt' for _ in cell_names[test_adt_idx]]

        rna_feature = rna_names
        adt_feature = adt_names

        single_rna = SingleData("GeneExpr", "Single-GEX", rna_feature, single_rna_count, single_rna_barcode)
        single_adt = SingleData("ADT", "Single-ADT", adt_feature, single_adt_count, single_adt_barcode)

        multi_dt = MultiomicDataset.from_singledata(
            single_rna, single_adt, mult_adt, mult_rna)
        print(multi_dt)

        # saving for reproducibility
        np.save(join(save_dir, f'p={p}_r={repeat}_new_train_idx.npy'), new_train_idx)
        np.save(join(save_dir, f'p={p}_r={repeat}_test_rna_idx.npy'), test_rna_idx)
        np.save(join(save_dir, f'p={p}_r={repeat}_test_adt_idx.npy'), test_adt_idx)

        ## training
        model = Cobolt(dataset=multi_dt, lr=0.0001, n_latent=10, batch_size=128) # lr=0.001, diverge problem
        model.train(num_epochs=100)
        model.calc_all_latent()
        latent = model.get_all_latent()

        latent_barcode = np.array([_.split('~')[1] for _ in latent[1]])
        df_latent = pd.DataFrame(latent[0], index=latent_barcode)

        gex_test = sc.AnnData(single_rna_count)
        gex_test.obs_names = single_rna_barcode
        other_test = sc.AnnData(single_adt_count)
        other_test.obs_names = single_adt_barcode

        ad_mult = sc.AnnData(mult_rna_count)
        ad_mult.obs_names = mult_barcode
        ad_mult.obsm['X_emb'] = df_latent.loc[ad_mult.obs_names, :].values
        gex_test.obsm['X_emb'] = df_latent.loc[gex_test.obs_names, :].values
        other_test.obsm['X_emb'] = df_latent.loc[other_test.obs_names, :].values

        ad_mult.obs[['batch', 'celltype.l1', 'celltype.l2']] = meta_data.loc[ad_mult.obs_names, ['batch', 'celltype.l1', 'celltype.l2']].to_numpy()
        gex_test.obs[['batch', 'celltype.l1', 'celltype.l2']] = meta_data.loc[
            [_.rsplit('_', 1)[0] for _ in gex_test.obs_names], 
            ['batch', 'celltype.l1', 'celltype.l2']
        ].to_numpy()
        other_test.obs[['batch', 'celltype.l1', 'celltype.l2']] = meta_data.loc[
            [_.rsplit('_', 1)[0] for _ in other_test.obs_names], 
            ['batch', 'celltype.l1', 'celltype.l2']
        ].to_numpy()

        ad_mult.obs['mod'] = 'multiome'
        gex_test.obs['mod'] = 'gex'
        other_test.obs['mod'] = 'other'
        ad_mult.obs['mod-batch'] = ad_mult.obs.batch.apply(lambda x: 'multiome'+'-'+x).to_numpy()
        gex_test.obs['mod-batch'] = gex_test.obs.batch.apply(lambda x: 'gex'+'-'+x).to_numpy()
        other_test.obs['mod-batch'] = other_test.obs.batch.apply(lambda x: 'other'+'-'+x).to_numpy()

        print('================================')
        print(f'p={p}, repeat={repeat}')
        print('================================')

        # without harmony
        print('===============Before harmony==============')
        ad_mosaic = sc.concat([ad_mult, gex_test, other_test])
        r = eval_mosaic(ad_mosaic, label_key='celltype.l2', lisi_keys=['mod'], use_nmi=False, use_gc=False, use_rep='X_emb')

        nmi, ari = eval_clustering(
            ad_mosaic, label_key='celltype.l2', cluster_key='cluster', resolutions=None, use_rep='X_emb',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))

        # with harmony
        print('===============After harmony==============')
        ad_mosaic_df = pd.DataFrame(ad_mosaic.obsm['X_emb'], index=ad_mosaic.obs_names)
        ad_mosaic_df['batch'] = ad_mosaic.obs['mod-batch'].to_numpy()
        ad_mosaic.obsm['X_emb_harmony'] = harmony([ad_mosaic_df])[0]

        r = eval_mosaic(ad_mosaic, label_key='celltype.l2', 
            lisi_keys=['mod'], use_rep='X_emb_harmony', use_nmi=False, use_gc=False)
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='celltype.l2', cluster_key='cluster', resolutions=None, use_rep='X_emb_harmony',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))