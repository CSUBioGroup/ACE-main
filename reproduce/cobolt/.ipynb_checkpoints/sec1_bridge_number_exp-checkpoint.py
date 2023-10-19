from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scanpy as sc

from os.path import join
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Path to the data directory
dat_dir = "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/"
data_dir = os.path.join(dat_dir, "match_modality/openproblems_bmmc_cite_phase2_mod2")

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
mod1_feature_type = set(input_train_mod1_0.var["feature_types"])
mod2_feature_type = set(input_train_mod2_0.var["feature_types"])

meta_dir = '/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main'
df_meta = pd.read_csv(os.path.join(meta_dir, 'output/datasets/cite_meta.csv'), index_col=0)
input_train_mod1_0.obs[['batch', 'cell_type']] = df_meta.loc[input_train_mod1_0.obs_names, ['batch', 'cell_type']].to_numpy()
input_train_mod2_0.obs[['batch', 'cell_type']] = df_meta.loc[input_train_mod2_0.obs_names, ['batch', 'cell_type']].to_numpy()


######### reducing bridge number here
num_range = np.arange(input_train_mod1_0.shape[0])
for del_size in [0.1, 0.2, 0.4, 0.8]:  # training propor: [0] + [0.9, 0.8, 0.6, 0.2]
    for repeat in range(3):
        select_idx, _ = train_test_split(num_range, test_size=del_size, stratify=input_train_mod1_0.obs.cell_type.to_numpy())
        np.save(
            f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/bridge_num/cite_DelSize={del_size}_r={repeat}_ids.npy', 
            input_train_mod1_0.obs_names[select_idx].to_numpy()
        )

        input_train_mod1 = input_train_mod1_0[select_idx,].copy()
        input_train_mod2 = input_train_mod2_0[select_idx,].copy()

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

        single_rna_count = input_test_mod2_0[:, hvg_names].layers["counts"].astype(np.float32)
        single_adt_count = input_test_mod1_0.layers["counts"].astype(np.float32)
        single_rna_count = sps.csr_matrix(single_rna_count)
        single_adt_count = sps.csr_matrix(single_adt_count)
        single_rna_barcode = input_test_mod2_0.obs_names.to_numpy()
        single_adt_barcode = input_test_mod1_0.obs_names.to_numpy()

        rna_feature = hvg_names
        adt_feature = input_test_mod1_0.var_names.to_numpy()
        single_rna = SingleData("GeneExpr", "Single-GEX", rna_feature, single_rna_count, single_rna_barcode)
        single_adt = SingleData("ADT", "Single-ADT", adt_feature, single_adt_count, single_adt_barcode)
        multi_dt = MultiomicDataset.from_singledata(
            single_rna, single_adt, mult_adt, mult_rna)

        model = Cobolt(dataset=multi_dt, lr=0.0001, n_latent=10, batch_size=128) # all default
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

        input_test_sol = sc.read_h5ad(join(data_dir, 'openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_sol.h5ad'))
        test_ord = input_test_sol.X.tocsr().indices
        assert (test_ord == np.argsort(input_test_sol.uns['pairing_ix'])).all()
        gex_test = gex_test[test_ord, :].copy()
        gex_test.obs_names = other_test.obs_names.to_numpy()

        ad_mult.obs[['batch', 'cell_type']] = df_meta.loc[ad_mult.obs_names, ['batch', 'cell_type']].to_numpy()
        gex_test.obs[['batch', 'cell_type']] = df_meta.loc[gex_test.obs_names, ['batch', 'cell_type']].to_numpy()
        other_test.obs[['batch', 'cell_type']] = df_meta.loc[other_test.obs_names, ['batch', 'cell_type']].to_numpy()

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
        print(f'del_size={del_size}, repeat={repeat}')
        print('================================')

        import sys
        sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/ACE/reproduce/evaluation')
        from evaluation import eval_mosaic, eval_specific_mod, eval_bridge, print_results, eval_asw, eval_lisi, eval_clustering

        # without harmony
        print('===============Before harmony==============')
        ad_mosaic = sc.concat([ad_mult, gex_test, other_test])
        r = eval_mosaic(ad_mosaic, label_key='cell_type', lisi_keys=['mod-batch', 'mod'], 
            use_rep='X_emb', use_gc=False, use_nmi=False)

        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_emb',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))

        r = eval_bridge(
            gex_test, other_test,
            label_key='cell_type',
            batch_key='batch',
            use_rep='X_emb',
            use_acc=False
        )

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

        gex_test.obsm['X_emb_harmony'] = ad_mosaic.obsm['X_emb_harmony'][ad_mult.shape[0]:(ad_mult.shape[0] + gex_test.shape[0])]
        other_test.obsm['X_emb_harmony'] = ad_mosaic.obsm['X_emb_harmony'][(-other_test.shape[0]):]
        r = eval_bridge(
                gex_test, other_test,
                label_key='cell_type',
                batch_key='batch',
                use_rep='X_emb_harmony',
                use_acc=False
        )