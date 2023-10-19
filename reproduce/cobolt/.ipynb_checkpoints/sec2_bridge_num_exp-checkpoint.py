from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os
import gc
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
from evaluation import eval_bridge_above2
from preprocessing import harmony

# Path to the data directory
data_dir = "/home/sda1/yanxh/data/DOGMA"

# reading data
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

train_idx = np.where((meta_data.stim=='Control').to_numpy())[0]
test_idx  = np.where((meta_data.stim=='Stim').to_numpy())[0]


######### reducing bridge number here
num_range = np.arange(train_idx.size)
for del_size in [0.1, 0.2, 0.4, 0.8]:  # training propor: [0] + [0.9, 0.8, 0.6, 0.2]
    for repeat in range(3):
        select_idx, _ = train_test_split(num_range, test_size=del_size, 
            stratify=meta_data.loc[cell_names[train_idx], 'predicted.celltype.l1'])
        np.save(
            f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/dogma_bridge_num/DelSize={del_size}_r={repeat}_ids.npy', 
            train_idx[select_idx]
        )
        new_train_idx = train_idx[select_idx]
        mult_rna_count = rna_count_mat[new_train_idx].astype(np.float32)
        mult_adt_count = adt_count_mat[new_train_idx].astype(np.float32)
        mult_atac_count = atac_count_mat[new_train_idx].astype(np.float32)
        mult_rna_count = sps.csr_matrix(mult_rna_count)   # this fucking thing must be in csr format, otherwise, training will be fucking very 
        mult_adt_count = sps.csr_matrix(mult_adt_count)   # slowly, fuck, fuck, fuck
        mult_atac_count = sps.csr_matrix(mult_atac_count) 

        mult_barcode = cell_names[new_train_idx]
        rna_feature = rna_names
        adt_feature = adt_names
        atac_feature = atac_names

        mult_rna = SingleData("GeneExpr", "Multiome", rna_feature, mult_rna_count, mult_barcode)
        mult_adt = SingleData("ADT", "Multiome", adt_feature, mult_adt_count, mult_barcode)
        mult_atac = SingleData("ATAC", "Multiome", atac_feature, mult_atac_count, mult_barcode)

        single_rna_count = rna_count_mat[test_idx].astype(np.float32)
        single_adt_count = adt_count_mat[test_idx].astype(np.float32)
        single_atac_count = atac_count_mat[test_idx].astype(np.float32)
        single_rna_count = sps.csr_matrix(single_rna_count)
        single_adt_count = sps.csr_matrix(single_adt_count)
        single_atac_count = sps.csr_matrix(single_atac_count)
        single_rna_barcode = [_+'_rna' for _ in cell_names[test_idx]]
        single_adt_barcode = [_+'_adt' for _ in cell_names[test_idx]]
        single_atac_barcode = [_+'_atac' for _ in cell_names[test_idx]]

        single_rna = SingleData("GeneExpr", "Single-GEX", rna_feature, single_rna_count, single_rna_barcode)
        single_adt = SingleData("ADT", "Single-ADT", adt_feature, single_adt_count, single_adt_barcode)
        single_atac = SingleData("ATAC", "Single-ATAC", atac_feature, single_atac_count, single_atac_barcode)

        multi_dt = MultiomicDataset.from_singledata(
            single_rna, single_adt, single_atac, mult_adt, mult_rna, mult_atac)
        print(multi_dt)

        model = Cobolt(dataset=multi_dt, lr=0.001, n_latent=10, batch_size=128) 
        model.train(num_epochs=100)
        model.calc_all_latent()
        latent = model.get_all_latent()
        latent_barcode = np.array([_.split('~')[1] for _ in latent[1]])
        df_latent = pd.DataFrame(latent[0], index=latent_barcode)

        gex_test = sc.AnnData(single_rna_count)
        gex_test.obs_names = single_rna_barcode
        adt_test = sc.AnnData(single_adt_count)
        adt_test.obs_names = single_adt_barcode
        atac_test = sc.AnnData(single_atac_count)
        atac_test.obs_names = single_atac_barcode

        ad_mult = sc.AnnData(mult_rna_count)
        ad_mult.obs_names = mult_barcode

        ad_mult.obsm['X_emb'] = df_latent.loc[ad_mult.obs_names, :].values
        gex_test.obsm['X_emb'] = df_latent.loc[gex_test.obs_names, :].values
        adt_test.obsm['X_emb'] = df_latent.loc[adt_test.obs_names, :].values
        atac_test.obsm['X_emb'] = df_latent.loc[atac_test.obs_names, :].values

        ad_mult.obs[['batch', 'cell_type']] = meta_data.loc[ad_mult.obs_names, ['stim', 'predicted.celltype.l1']].to_numpy()
        gex_test.obs[['batch', 'cell_type']] = meta_data.loc[
            [_.rsplit('_', 1)[0] for _ in gex_test.obs_names], 
            ['stim', 'predicted.celltype.l1']
        ].to_numpy()
        adt_test.obs[['batch', 'cell_type']] = meta_data.loc[
            [_.rsplit('_', 1)[0] for _ in adt_test.obs_names], 
            ['stim', 'predicted.celltype.l1']
        ].to_numpy()
        atac_test.obs[['batch', 'cell_type']] = meta_data.loc[
            [_.rsplit('_', 1)[0] for _ in atac_test.obs_names], 
            ['stim', 'predicted.celltype.l1']
        ].to_numpy()

        ad_mult.obs['mod'] = 'multiome'
        gex_test.obs['mod'] = 'gex'
        adt_test.obs['mod'] = 'adt'
        atac_test.obs['mod'] = 'atac'
        ad_mult.uns['mod'] = 'multiome'
        gex_test.uns['mod'] = 'gex'
        adt_test.uns['mod'] = 'adt'
        atac_test.uns['mod'] = 'atac'

        ad_mult.obs['mod-batch'] = ad_mult.obs.batch.apply(lambda x: 'multiome'+'-'+x).to_numpy()
        gex_test.obs['mod-batch'] = gex_test.obs.batch.apply(lambda x: 'gex'+'-'+x).to_numpy()
        adt_test.obs['mod-batch'] = adt_test.obs.batch.apply(lambda x: 'adt'+'-'+x).to_numpy()
        atac_test.obs['mod-batch'] = atac_test.obs.batch.apply(lambda x: 'atac'+'-'+x).to_numpy()

        ############################
        # evaluation
        ############################
        print('=========================')
        print(f'del_size={del_size}, repeat={repeat}')
        print('=========================')
        print('===============Before harmony==============')
        # without harmony
        ad_mosaic = sc.concat([ad_mult, gex_test, adt_test, atac_test])

        r = eval_mosaic(ad_mosaic, label_key='cell_type', lisi_keys=['mod-batch'], use_rep='X_emb',
                       use_lisi=True, use_gc=False, use_nmi=False)
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))
        r = eval_bridge_above2(
            [gex_test, adt_test, atac_test],
            label_key='cell_type',
            batch_key='batch',
            mod_key='mod',
            use_rep='X_emb',
            use_acc=False
        )

        print('===============After harmony==============')
        # with harmony
        ad_mosaic_df = pd.DataFrame(ad_mosaic.obsm['X_emb'], index=ad_mosaic.obs_names)
        ad_mosaic_df['batch'] = ad_mosaic.obs['mod-batch'].to_numpy()
        ad_mosaic.obsm['X_emb_harmony'] = harmony([ad_mosaic_df])[0]

        r = eval_mosaic(ad_mosaic, label_key='cell_type', lisi_keys=['mod-batch'], use_rep='X_emb_harmony',
                       use_lisi=True, use_gc=False, use_nmi=False)

        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb_harmony',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))

        gex_test.obsm['X_emb_harmony'] = ad_mosaic.obsm['X_emb_harmony'][ad_mult.shape[0]:(ad_mult.shape[0] + gex_test.shape[0])]
        adt_test.obsm['X_emb_harmony'] = ad_mosaic.obsm['X_emb_harmony'][
            (ad_mult.shape[0] + gex_test.shape[0]):(ad_mult.shape[0] + gex_test.shape[0] + adt_test.shape[0])]
        atac_test.obsm['X_emb_harmony'] = ad_mosaic.obsm['X_emb_harmony'][-atac_test.shape[0]:]

        r = eval_bridge_above2(
            [gex_test, adt_test, atac_test],
            label_key='cell_type',
            batch_key='batch',
            mod_key='mod',
            use_rep='X_emb_harmony',
            use_acc=False
        )