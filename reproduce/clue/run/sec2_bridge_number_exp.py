import gc
import logging
import os
import pickle
import sys
import anndata as ad
import numpy as np
import pandas as pd
import yaml
import sys
import scanpy as sc
import scipy.sparse as sps
import scipy.io as sio

import scglue
import seaborn as sns

from os.path import join

import matplotlib.pyplot as plt
# Path to the data directory
root_dir = '/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main'
data_dir = "/home/sda1/yanxh/data/DOGMA"

sys.path.append(os.path.join(root_dir, 'src/match_modality/methods/clue/resources'))
import utils
sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/ACE/reproduce/evaluation')
from preprocessing import harmony
from evaluation import eval_mosaic, eval_specific_mod, eval_bridge, print_results, eval_asw, eval_lisi, eval_clustering
from evaluation import eval_bridge_above2

par = {}
par['output_pretrain'] = os.path.join(
    root_dir, 
    'output/pretrain/clue/dogma_bridge_num/')

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
meta_data['batch'] = meta_data.stim.to_numpy()

train_idx = np.where((meta_data.batch=='Control').to_numpy())[0]
test_idx  = np.where((meta_data.batch=='Stim').to_numpy())[0]


mod1_feature_type = 'GEX'
mod2_feature_type = 'ADT'
mod3_feature_type = 'ATAC'
omics = 'cite'

n_genes = 5000
latent_dim = 20
x2u_h_depth = 2
x2u_h_dim = 512
u2x_h_depth = 1
u2x_h_dim = 128
du_h_depth = 2
du_h_dim = 128
dropout = 0.2
lam_data = 1.0
lam_kl = 1.0
lam_align = 2.0
lam_cross = 2.0
lam_cos = 1.0
normalize_u = True
random_seed = 5

for del_size in [0.1, 0.2, 0.4, 0.8]:  
    for repeat in range(3):
        new_train_idx =  np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/dogma_bridge_num/DelSize={del_size}_r={repeat}_ids.npy')
        save_dir = os.path.join(par['output_pretrain'], f'DelSize={del_size}-repeat={repeat}')
        os.makedirs(save_dir, exist_ok=True)

        print('Reading `h5ad` files...')
        ad_mult_rna = sc.AnnData(sps.csr_matrix(rna_count_mat[new_train_idx]), obs=meta_data.iloc[new_train_idx])
        ad_mult_adt = sc.AnnData(sps.csr_matrix(adt_count_mat[new_train_idx]), obs=meta_data.iloc[new_train_idx])
        ad_mult_atac = sc.AnnData(sps.csr_matrix(atac_count_mat[new_train_idx]), obs=meta_data.iloc[new_train_idx])

        ad_rna_test = sc.AnnData(sps.csr_matrix(rna_count_mat[test_idx]), obs=meta_data.iloc[test_idx])
        ad_adt_test = sc.AnnData(sps.csr_matrix(adt_count_mat[test_idx]), obs=meta_data.iloc[test_idx])
        ad_atac_test = sc.AnnData(sps.csr_matrix(atac_count_mat[test_idx]), obs=meta_data.iloc[test_idx])

        ad_mult_rna.var_names = ad_rna_test.var_names = rna_names
        ad_mult_adt.var_names = ad_adt_test.var_names= adt_names
        ad_mult_atac.var_names = ad_atac_test.var_names = atac_names

        ad_mult_rna.layers["counts"] = ad_mult_rna.X.astype(np.float32)
        ad_mult_adt.layers["counts"] = ad_mult_adt.X.astype(np.float32)
        ad_mult_atac.layers["counts"] = ad_mult_atac.X.astype(np.float32)
        ad_mult_rna.layers["counts"] = ad_mult_rna.X.astype(np.float32)
        ad_mult_adt.layers["counts"] = ad_mult_adt.X.astype(np.float32)
        ad_mult_atac.layers["counts"] = ad_mult_atac.X.astype(np.float32)
        ad_rna_test.layers["counts"] = ad_rna_test.X.astype(np.float32)
        ad_adt_test.layers["counts"] = ad_adt_test.X.astype(np.float32)
        ad_atac_test.layers["counts"] = ad_atac_test.X.astype(np.float32)

        with open(os.path.join(save_dir, "hyperparams.yaml"), "w") as f:
            yaml.dump({
                "n_genes": n_genes,
                "latent_dim": latent_dim,
                "x2u_h_depth": x2u_h_depth,
                "x2u_h_dim": x2u_h_dim,
                "u2x_h_depth": u2x_h_depth,
                "u2x_h_dim": u2x_h_dim,
                "du_h_depth": du_h_depth,
                "du_h_dim": du_h_dim,
                "dropout": dropout,
                "lam_data": lam_data,
                "lam_kl": lam_kl,
                "lam_align": lam_align,
                "lam_cross": lam_cross,
                "lam_cos": lam_cos,
                "normalize_u": normalize_u,
                "random_seed": random_seed
            }, f)

        ad_mult_rna.obs["uid"] = [f"train-{i}" for i in range(ad_mult_rna.shape[0])]
        ad_mult_adt.obs["uid"] = [f"train-{i}" for i in range(ad_mult_adt.shape[0])]
        ad_mult_atac.obs["uid"] = [f"train-{i}" for i in range(ad_mult_atac.shape[0])]

        ad_mult_rna.obs['domain'] = 'gex'
        ad_mult_adt.obs['domain'] = 'adt'
        ad_mult_atac.obs['domain'] = 'atac'
        ad_mult_rna.uns['domain'] = 'gex'
        ad_mult_adt.uns['domain'] = 'adt'
        ad_mult_atac.uns['domain'] = 'atac'

        print('Preprocessing GEX...')
        gex_prep = utils.GEXPreprocessing(n_comps=100, n_genes=n_genes, merge_adt=False)
        gex_prep.fit_transform(ad_mult_rna)

        print('Preprocessing ADT...')
        adt_prep = utils.ADTPreprocessing(n_comps=100)

        print('Preprocessing ATAC...')
        atac_prep = utils.ATACPreprocessing(n_comps=100)
            
        adt_prep.fit_transform(ad_mult_adt)
        atac_prep.fit_transform(ad_mult_atac)

        with open(os.path.join(save_dir, "prep.pickle"), "wb") as f:
            pickle.dump({
                "gex_prep": gex_prep,
                "adt_prep": adt_prep,
                "atac_prep": atac_prep,
            }, f)

        scglue.models.configure_dataset(
            ad_mult_rna, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            use_batch="stim", use_uid="uid"
        )
        scglue.models.configure_dataset(
            ad_mult_adt, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            use_batch="stim", use_uid="uid"
        )
        scglue.models.configure_dataset(
            ad_mult_atac, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_lsi",
            use_batch="stim", use_uid="uid"
        )

        print('Building model...')
        model = scglue.models.SCCLUEModel(
            {"gex": ad_mult_rna, "adt": ad_mult_adt, "atac":ad_mult_atac},
            latent_dim=latent_dim,
            x2u_h_depth=x2u_h_depth,
            x2u_h_dim=x2u_h_dim,
            u2x_h_depth=u2x_h_depth,
            u2x_h_dim=u2x_h_dim,
            du_h_depth=du_h_depth,
            du_h_dim=du_h_dim,
            dropout=dropout,
            shared_batches=True,
            random_seed=random_seed
        )

        # training = True
        print('Compiling model...')
        model.compile(
            lam_data=lam_data, lam_kl=lam_kl, lam_align=lam_align,
            lam_cross=lam_cross, lam_cos=lam_cos, normalize_u=normalize_u,
            domain_weight={"gex": 1, "adt": 1, 'atac':1}
        )

        if not os.path.exists(os.path.join(save_dir, 'pretrain.dill')):
            print('Training model...')
            model.fit(
                {"gex": ad_mult_rna, "adt": ad_mult_adt, "atac":ad_mult_atac}
            )
            model.save(os.path.join(save_dir, "pretrain.dill"))
        else:
            # loading pretrained weight
            model = scglue.models.load_model(os.path.join(save_dir, "pretrain.dill"))

        ##### stage 2

        # shuffling test data
        test_shuffle_idx1 = np.arange(ad_rna_test.shape[0])
        np.random.shuffle(test_shuffle_idx1)
        test_reorder_idx1 = np.argsort(test_shuffle_idx1)

        test_shuffle_idx2 = np.arange(ad_adt_test.shape[0])
        np.random.shuffle(test_shuffle_idx2)
        test_reorder_idx2 = np.argsort(test_shuffle_idx2)

        test_shuffle_idx3 = np.arange(ad_atac_test.shape[0])
        np.random.shuffle(test_shuffle_idx3)
        test_reorder_idx3 = np.argsort(test_shuffle_idx3)

        ad_rna_test = ad_rna_test[test_shuffle_idx1].copy()
        ad_adt_test = ad_adt_test[test_shuffle_idx2].copy()
        ad_atac_test = ad_atac_test[test_shuffle_idx3].copy()

        ad_mult_rna.obs["uid"] = [f"train-{i}" for i in range(ad_mult_rna.shape[0])]
        ad_mult_adt.obs["uid"] = [f"train-{i}" for i in range(ad_mult_adt.shape[0])]
        ad_mult_atac.obs["uid"] = [f"train-{i}" for i in range(ad_mult_atac.shape[0])]
        ad_rna_test.obs["uid"] = [f"test-gex-{i}" for i in range(ad_rna_test.shape[0])]
        ad_adt_test.obs["uid"] = [f"test-adt-{i}" for i in range(ad_adt_test.shape[0])]
        ad_atac_test.obs["uid"] = [f"test-atac-{i}" for i in range(ad_atac_test.shape[0])]
        ad_rna_test.obs_names = [f"rna-{i}" for i in ad_rna_test.obs_names.to_numpy()]
        ad_adt_test.obs_names = [f"adt-{i}" for i in ad_adt_test.obs_names.to_numpy()]
        ad_atac_test.obs_names = [f"atac-{i}" for i in ad_atac_test.obs_names.to_numpy()]

        def set_domain(ads, obs_key, uns_key, domains):
            for i in range(len(ads)):
                ads[i].obs[obs_key] = domains[i]
                ads[i].uns[uns_key] = domains[i]
            return ads

        logging.info('Concatenating training and test data...')
        input_gex = ad.concat(
            {"train": ad_mult_rna, "test": ad_rna_test},
            axis=0, join="outer", merge="same", label="group",
            fill_value=0, index_unique="-"
        )
        input_gex.uns["feature_type"] = 'GEX'
        del ad_mult_rna, ad_rna_test
        gc.collect()

        input_adt = ad.concat(
            {"train": ad_mult_adt, "test": ad_adt_test},
            axis=0, join="outer", merge="same", label="group",
            fill_value=0, index_unique="-"
        )
        input_adt.uns["feature_type"] = 'ADT'
        del ad_mult_adt, ad_adt_test
        gc.collect()

        input_atac = ad.concat(
            {"train": ad_mult_atac, "test": ad_atac_test},
            axis=0, join="outer", merge="same", label="group",
            fill_value=0, index_unique="-"
        )
        input_atac.uns["feature_type"] = 'ATAC'
        del ad_mult_atac, ad_atac_test
        gc.collect()

        logging.info('Reading preprocessors...')
        with open(os.path.join(
                save_dir, "prep.pickle"
        ), "rb") as f:
            prep = pickle.load(f)
            gex_prep = prep["gex_prep"]
            adt_prep = prep["adt_prep"]
            atac_prep = prep['atac_prep']

        logging.info('Preprocessing...')
        gex_prep.transform(input_gex)
        adt_prep.transform(input_adt)
        atac_prep.transform(input_atac)

        scglue.models.configure_dataset(
            input_gex, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_uid="uid"
        )
        scglue.models.configure_dataset(
            input_adt, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_uid="uid"
        )
        scglue.models.configure_dataset(
            input_atac, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_lsi",
            use_batch="batch", use_uid="uid"
        )

        with open(os.path.join(
                save_dir, "hyperparams.yaml"
        ), "r") as f:
            hyperparams = yaml.load(f, Loader=yaml.Loader)

        logging.info('Building model...')
        model = scglue.models.SCCLUEModel(
            {"gex": input_gex, "adt": input_adt, "atac":input_atac},
            latent_dim=hyperparams["latent_dim"],
            x2u_h_depth=hyperparams["x2u_h_depth"],
            x2u_h_dim=hyperparams["x2u_h_dim"],
            u2x_h_depth=hyperparams["u2x_h_depth"],
            u2x_h_dim=hyperparams["u2x_h_dim"],
            du_h_depth=hyperparams["du_h_depth"],
            du_h_dim=hyperparams["du_h_dim"],
            dropout=hyperparams["dropout"],
            shared_batches=True,
            random_seed=hyperparams["random_seed"]
        )

        logging.info('Adopting pretrained weights...')
        model.adopt_pretrained_model(scglue.models.load_model(os.path.join(
            save_dir, "pretrain.dill"
        )))
        logging.info('Compiling model...')
        model.compile(
            lam_data=hyperparams["lam_data"],
            lam_kl=hyperparams["lam_kl"],
            lam_align=hyperparams["lam_align"],
            lam_cross=hyperparams["lam_cross"],
            lam_cos=hyperparams["lam_cos"],
            normalize_u=hyperparams["normalize_u"],
            domain_weight={"gex": 1, "adt": 1, 'atac':1},
            lr=1e-3  # TODO: Fine-tuning learning rate
        )

        if not os.path.exists(os.path.join(save_dir, 'train.dill')):
            logging.info('Training model...')
            model.fit(
                {"gex": input_gex, "adt": input_adt, "atac":input_atac},
                align_burnin=0, max_epochs=50,  # 50 or 5
                patience=8, reduce_lr_patience=3
            )

            model.save(os.path.join(save_dir, "train.dill"))
        else:
            model.adopt_pretrained_model(scglue.models.load_model(os.path.join(
                save_dir, "train.dill"
            )))


        gex_enc = model.encode_data('gex', input_gex[input_gex.obs["group"] == "test"])
        adt_enc = model.encode_data('adt', input_adt[input_adt.obs["group"] == "test"])
        atac_enc = model.encode_data('atac', input_atac[input_atac.obs["group"] == "test"])

        gex_train_enc = model.encode_data('gex', input_gex[input_gex.obs["group"] == "train"])
        adt_train_enc = model.encode_data('adt', input_adt[input_adt.obs["group"] == "train"])
        atac_train_enc = model.encode_data('atac', input_atac[input_atac.obs["group"] == "train"])

        ad_rna_test = input_gex[input_gex.obs["group"] == "test"].copy()
        ad_adt_test = input_adt[input_adt.obs["group"] == "test"].copy()
        ad_atac_test = input_atac[input_atac.obs["group"] == "test"].copy()
        ad_mult_rna = input_gex[input_gex.obs["group"] == "train"].copy()
        ad_mult_adt = input_adt[input_adt.obs["group"] == "train"].copy()
        ad_mult_atac = input_atac[input_atac.obs["group"] == "train"].copy()

        ad_rna_test.obsm['X_model'] = gex_enc
        ad_adt_test.obsm['X_model'] = adt_enc
        ad_atac_test.obsm['X_model'] = atac_enc

        ad_mult_rna.obsm['X_model'] = gex_train_enc
        ad_mult_adt.obsm['X_model'] = adt_train_enc
        ad_mult_atac.obsm['X_model'] = atac_train_enc

        ad_mult_rna, ad_rna_test = set_domain([ad_mult_rna, ad_rna_test], obs_key='domain', uns_key='domain', domains=['multiome', 'gex'])
        ad_mult_adt, ad_adt_test = set_domain([ad_mult_adt, ad_adt_test], obs_key='domain', uns_key='domain', domains=['multiome', 'adt'])
        ad_mult_atac, ad_atac_test = set_domain([ad_mult_atac, ad_atac_test], obs_key='domain', uns_key='domain', domains=['multiome', 'atac'])

        ad_mult_rna.obs['mod-batch'] = (ad_mult_rna.obs.domain + '-' + ad_mult_rna.obs.batch).to_numpy()
        ad_mult_adt.obs['mod-batch'] = (ad_mult_adt.obs.domain + '-' + ad_mult_adt.obs.batch).to_numpy()
        ad_mult_atac.obs['mod-batch'] = (ad_mult_atac.obs.domain + '-' + ad_mult_atac.obs.batch).to_numpy()
        ad_rna_test.obs['mod-batch'] = (ad_rna_test.obs.domain + '-' + ad_rna_test.obs.batch).to_numpy()
        ad_adt_test.obs['mod-batch'] = (ad_adt_test.obs.domain + '-' + ad_adt_test.obs.batch).to_numpy()
        ad_atac_test.obs['mod-batch'] = (ad_atac_test.obs.domain + '-' + ad_atac_test.obs.batch).to_numpy()

        ad_rna_test = ad_rna_test[test_reorder_idx1].copy()
        ad_adt_test = ad_adt_test[test_reorder_idx2].copy()
        ad_atac_test = ad_atac_test[test_reorder_idx3].copy()

        rna_test_ids = np.array([_.split("-", 1)[1] for _ in ad_rna_test.obs_names])
        adt_test_ids = np.array([_.split("-", 1)[1] for _ in ad_adt_test.obs_names])
        atac_test_ids = np.array([_.split("-", 1)[1] for _ in ad_atac_test.obs_names])

        assert (rna_test_ids==adt_test_ids).all()
        assert (rna_test_ids==atac_test_ids).all()
        assert (ad_rna_test.obs.batch.to_numpy() == ad_adt_test.obs.batch.to_numpy()).all()
        assert (ad_rna_test.obs.batch.to_numpy() == ad_atac_test.obs.batch.to_numpy()).all()

        #### evaluation
        print('=========================')
        print(f'del_size={del_size}, repeat={repeat}')
        print('=========================')
        print('===============Before harmony==============')
        # without harmony
        ad_train_mean_enc = sc.AnnData(
            (ad_mult_rna.obsm['X_model'] + ad_mult_adt.obsm['X_model'] + ad_mult_atac.obsm['X_model'])/3, 
            obs=ad_mult_rna.obs.copy()
        )
        ad_train_mean_enc.obsm['X_model'] = ad_train_mean_enc.X.copy()
        ad_mosaic = sc.concat([ad_train_mean_enc, ad_rna_test, ad_adt_test, ad_atac_test])

        r = eval_mosaic(ad_mosaic, label_key='predicted.celltype.l1', 
                        lisi_keys=['mod-batch'], use_rep='X_model', use_lisi=True, use_gc=False, use_nmi=False)
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='predicted.celltype.l1', cluster_key='cluster', resolutions=None, use_rep='X_model',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))

        r = eval_bridge_above2(
                [ad_rna_test, ad_adt_test, ad_atac_test],
                label_key='predicted.celltype.l1',
                batch_key='batch',
                mod_key='domain',
                use_rep='X_model',
                use_acc=False
        )

        print('===============After harmony==============')
        ad_mosaic_df = pd.DataFrame(ad_mosaic.obsm['X_model'], index=ad_mosaic.obs_names.to_numpy())
        ad_mosaic_df['batch'] = ad_mosaic.obs['mod-batch'].to_numpy()
        ad_mosaic.obsm['X_model_harmony'] = harmony([ad_mosaic_df])[0]

        r = eval_mosaic(ad_mosaic, label_key='predicted.celltype.l1', lisi_keys=['mod-batch'], use_rep='X_model_harmony', 
                        use_lisi=True, use_gc=False, use_nmi=False)
        nmi, ari = eval_clustering(
            ad_mosaic, label_key='predicted.celltype.l1', cluster_key='cluster', resolutions=None, use_rep='X_model_harmony',
            use='nmi', nmi_method='arithmetic')
        print('nmi={:.4f}, ari={:.4f}'.format(nmi, ari))

        ad_rna_test.obsm['X_model_harmony'] = ad_mosaic.obsm['X_model_harmony'][ad_mult_rna.shape[0]:(ad_mult_rna.shape[0] + ad_rna_test.shape[0])]
        ad_adt_test.obsm['X_model_harmony'] = ad_mosaic.obsm['X_model_harmony'][
            (ad_mult_rna.shape[0] + ad_rna_test.shape[0]):(ad_mult_rna.shape[0] + ad_rna_test.shape[0] + ad_adt_test.shape[0])
        ]
        ad_atac_test.obsm['X_model_harmony'] = ad_mosaic.obsm['X_model_harmony'][(-ad_atac_test.shape[0]):]

        r = eval_bridge_above2(
                [ad_rna_test, ad_adt_test, ad_atac_test],
                label_key='predicted.celltype.l1',
                batch_key='batch',
                mod_key='domain',
                use_rep='X_model_harmony',
                use_acc=False
        )

