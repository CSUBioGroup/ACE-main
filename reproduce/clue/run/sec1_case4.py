import gc
import logging
import os
import pickle

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
sys.path.append(os.path.join(root_dir, 'src/match_modality/methods/clue/resources'))
import utils

sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/ACE/reproduce/evaluation')
from evaluation import eval_mosaic, eval_specific_mod, eval_bridge, print_results, eval_asw, eval_lisi, eval_clustering
from preprocessing import harmony

input_dir = '/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/cite'
dataset_path = os.path.join(root_dir, 'output/datasets/match_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_')
par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'input_train_sol': f'{dataset_path}train_sol.h5ad',
    'input_test_sol': f'{dataset_path}test_sol.h5ad',
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
    'output_pretrain': os.path.join(root_dir, 
        'output/pretrain/clue/case4_cite')
}
meta = { 'resources_dir': os.path.join(root_dir, 'src/match_modality/methods/clue/resources') }


#######################################
#  stage 1
#######################################
print('Reading `h5ad` files...')
logging.info('Reading `h5ad` files...')
input_train_mod1_0 = sc.read_h5ad(par['input_train_mod1'])
input_train_mod2_0 = sc.read_h5ad(par['input_train_mod2'])
input_train_sol = sc.read_h5ad(par['input_train_sol'])
input_test_sol = sc.read_h5ad(par['input_test_sol'])
input_test_mod1_0 = sc.read_h5ad(par['input_test_mod1'])
input_test_mod2_0 = sc.read_h5ad(par['input_test_mod2'])

input_train_mod1_0.X = input_train_mod1_0.X.astype(np.float32)
input_train_mod2_0.X = input_train_mod2_0.X.astype(np.float32)
input_test_mod1_0.X = input_test_mod1_0.X.astype(np.float32)
input_test_mod2_0.X = input_test_mod2_0.X.astype(np.float32)
input_train_mod1_0.layers["counts"] = input_train_mod1_0.layers["counts"].astype(np.float32)
input_train_mod2_0.layers["counts"] = input_train_mod2_0.layers["counts"].astype(np.float32)
input_test_mod1_0.layers["counts"] = input_test_mod1_0.layers["counts"].astype(np.float32)
input_test_mod2_0.layers["counts"] = input_test_mod2_0.layers["counts"].astype(np.float32)

dataset_id = {
    input_train_mod1_0.uns["dataset_id"],
    input_train_mod2_0.uns["dataset_id"],
    input_test_mod1_0.uns["dataset_id"],
    input_test_mod2_0.uns["dataset_id"]
}
assert len(dataset_id) == 1
dataset_id = dataset_id.pop()

logging.info("Unscrambling training cells...")
ord = input_train_sol.X.tocsr().indices
train_mod2_ord = ord.copy()
if "pairing_ix" in input_train_sol.uns:
    assert np.all(ord == np.argsort(input_train_sol.uns["pairing_ix"]))
input_train_mod2_0 = input_train_mod2_0[ord, :].copy()
input_train_mod2_0.obs_names = input_train_mod1_0.obs_names
assert np.all(input_train_mod1_0.obs["batch"] == input_train_mod2_0.obs["batch"])

df_meta = pd.read_csv(os.path.join(root_dir, 'output/datasets/cite_meta.csv'), index_col=0)
input_train_mod1_0.obs['cell_type'] = df_meta.loc[input_train_mod1_0.obs_names, 'cell_type'].to_numpy()
input_train_mod2_0.obs['cell_type'] = input_train_mod1_0.obs.cell_type.to_numpy()
input_test_mod1_0.obs['cell_type']  = df_meta.loc[input_test_mod1_0.obs_names, 'cell_type'].to_numpy()

test_ord = input_test_sol.X.tocsr().indices
assert (test_ord == np.argsort(input_test_sol.uns['pairing_ix'])).all()
input_test_mod2_0.obs['cell_type']  = input_test_mod1_0.obs.cell_type.to_numpy()[np.argsort(test_ord)]

mod1_feature_type = set(input_train_mod1_0.var["feature_types"])
mod2_feature_type = set(input_train_mod2_0.var["feature_types"])
assert len(mod1_feature_type) == len(mod2_feature_type) == 1
mod1_feature_type = mod1_feature_type.pop()
mod2_feature_type = mod2_feature_type.pop()

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

omics = 'cite'

for p in [0.1, 0.2, 0.4, 0.8]:
    for r in range(3):
        new_train_idx = np.load(join(input_dir, f'p={p}_r={r}_new_train_idx.npy'))
        test_rna_idx  = np.load(join(input_dir, f'p={p}_r={r}_test_rna_idx.npy'))
        test_adt_idx  = np.load(join(input_dir, f'p={p}_r={r}_test_adt_idx.npy'))
        save_dir = join(par['output_pretrain'], f'p={p}_r={r}')
        os.makedirs(save_dir, exist_ok=True)

        print('Reading `h5ad` files...')
        input_train_mod1 = input_train_mod1_0[new_train_idx].copy()
        input_train_mod2 = input_train_mod2_0[new_train_idx].copy()
        input_train_mod1.obs["uid"] = [f"train-{i}" for i in range(input_train_mod1.shape[0])]
        input_train_mod2.obs["uid"] = [f"train-{i}" for i in range(input_train_mod2.shape[0])]

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

        if mod1_feature_type == "GEX":
            gex = input_train_mod1
            other = input_train_mod2
        else:
            gex = input_train_mod2
            other = input_train_mod1

        print('Preprocessing GEX...')
        gex_prep = utils.GEXPreprocessing(n_comps=100, n_genes=n_genes, merge_adt=omics == "cite")
        gex_prep.fit_transform(gex)

        if omics == "cite":
            print('Preprocessing ADT...')
            other_prep = utils.ADTPreprocessing(n_comps=100)
        elif omics == "multiome":
            print('Preprocessing ATAC...')
            other_prep = utils.ATACPreprocessing(n_comps=100)
        other_prep.fit_transform(other)

        with open(os.path.join(save_dir, "prep.pickle"), "wb") as f:
            pickle.dump({
                "gex_prep": gex_prep,
                "other_prep": other_prep
            }, f)

        scglue.models.configure_dataset(
            gex, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_uid="uid"
        )
        scglue.models.configure_dataset(
            other, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_uid="uid"
        )

        print('Building model...')
        model = scglue.models.SCCLUEModel(
            {"gex": gex, "other": other},
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

        print('Compiling model...')
        model.compile(
            lam_data=lam_data, lam_kl=lam_kl, lam_align=lam_align,
            lam_cross=lam_cross, lam_cos=lam_cos, normalize_u=normalize_u,
            domain_weight={"gex": 1, "other": 1}
        )

        if not os.path.exists(os.path.join(save_dir, 'pretrain.dill')):
            print('Training model...')
            model.fit(
                {"gex": gex, "other": other}
            )
            model.save(os.path.join(save_dir, "pretrain.dill"))
        else:
            pass

        # =========================
        ###### running
        # =========================
        input_test_mod1 = input_test_mod1_0[test_adt_idx].copy()
        input_test_mod2 = input_test_mod2_0[test_rna_idx].copy()
        input_test_mod1.obs["uid"] = [f"test-mod1-{i}" for i in range(input_test_mod1.shape[0])]
        input_test_mod2.obs["uid"] = [f"test-mod2-{i}" for i in range(input_test_mod2.shape[0])]

        logging.info('Concatenating training and test data...')
        input_mod1 = ad.concat(
            {"train": input_train_mod1, "test": input_test_mod1},
            axis=0, join="outer", merge="same", label="group",
            fill_value=0, index_unique="-"
        )
        input_mod1.uns["feature_type"] = mod1_feature_type
        del input_train_mod1, input_test_mod1
        gc.collect()
        input_mod2 = ad.concat(
            {"train": input_train_mod2, "test": input_test_mod2},
            axis=0, join="outer", merge="same", label="group",
            fill_value=0, index_unique="-"
        )
        input_mod2.uns["feature_type"] = mod2_feature_type
        del input_train_mod2, input_test_mod2
        gc.collect()

        if mod1_feature_type == "GEX":
            gex, other = input_mod1, input_mod2
        elif mod2_feature_type == "GEX":
            gex, other = input_mod2, input_mod1

        logging.info('Reading preprocessors...')
        with open(os.path.join(
                save_dir, "prep.pickle"
        ), "rb") as f:
            prep = pickle.load(f)
            gex_prep = prep["gex_prep"]
            other_prep = prep["other_prep"]

        logging.info('Preprocessing...')
        gex_prep.transform(gex)
        other_prep.transform(other)

        # configure datasets
        logging.info('Fine-tuning model...')
        scglue.models.configure_dataset(
            gex, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            use_batch="batch", use_uid="uid"
        )
        scglue.models.configure_dataset(
            other, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_lsi" if other.uns["feature_type"] == "ATAC" else "X_pca",
            use_batch="batch", use_uid="uid"
        )

        with open(os.path.join(
                save_dir, "hyperparams.yaml"
        ), "r") as f:
            hyperparams = yaml.load(f, Loader=yaml.Loader)

        logging.info('Building model...')
        model = scglue.models.SCCLUEModel(
            {"gex": gex, "other": other},
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
        print(model.net)

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
            domain_weight={"gex": 1, "other": 1},
            lr=1e-3  # TODO: Fine-tuning learning rate
        )
        
        if not os.path.exists(os.path.join(save_dir, 'train.dill')):
            logging.info('Training model...')
            model.fit(
                {"gex": gex, "other": other},
                align_burnin=0, max_epochs=50,  # 50 or 5
                patience=8, reduce_lr_patience=3
            )

            model.save(os.path.join(save_dir, "train.dill"))
        else:
            model.adopt_pretrained_model(scglue.models.load_model(os.path.join(
                save_dir, "train.dill"
            )))

        ###### evaluation
        mod1_enc = model.encode_data('gex' if mod1_feature_type=='GEX' else 'other', input_mod1[input_mod1.obs["group"] == "test"])
        mod2_enc = model.encode_data('gex' if mod2_feature_type=='GEX' else 'other', input_mod2[input_mod2.obs["group"] == "test"])

        mod1_train_enc = model.encode_data('gex' if mod1_feature_type=='GEX' else 'other', input_mod1[input_mod1.obs["group"] == "train"])
        mod2_train_enc = model.encode_data('gex' if mod2_feature_type=='GEX' else 'other', input_mod2[input_mod2.obs["group"] == "train"])

        input_test_mod1 = input_mod1[input_mod1.obs["group"] == "test"].copy()
        input_test_mod2 = input_mod2[input_mod2.obs["group"] == "test"].copy()
        input_train_mod1 = input_mod1[input_mod1.obs["group"] == "train"].copy()
        input_train_mod2 = input_mod2[input_mod2.obs["group"] == "train"].copy()

        input_test_mod1.obsm['X_model'] = mod1_enc
        input_test_mod2.obsm['X_model'] = mod2_enc
        input_train_mod1.obsm['X_model'] = mod1_train_enc
        input_train_mod2.obsm['X_model'] = mod2_train_enc

        input_train_mod1.obs['domain'] = 'multiome'; input_test_mod1.obs['domain'] = 'GEX';
        input_train_mod2.obs['domain'] = 'multiome'; input_test_mod2.obs['domain'] = 'ADT';

        input_train_mod1.obs['mod-batch'] = input_train_mod1.obs.batch.apply(lambda x: 'multiome'+'-'+x).to_numpy()
        input_train_mod2.obs['mod-batch'] = input_train_mod2.obs.batch.apply(lambda x: 'multiome'+'-'+x).to_numpy()
        input_test_mod1.obs['mod-batch'] = input_test_mod1.obs.batch.apply(lambda x: mod1_feature_type+'-'+x).to_numpy()
        input_test_mod2.obs['mod-batch'] = input_test_mod2.obs.batch.apply(lambda x: mod2_feature_type+'-'+x).to_numpy()

        ############################
        # evaluation
        ############################
        print('================================')
        print(f'p={p}, repeat={r}')
        print('================================')

        # without harmony
        print('=====================before harmony=====================')
        ad_train_mean_enc = sc.AnnData(
            (input_train_mod1.obsm['X_model'] + input_train_mod2.obsm['X_model'])/2, 
            obs=input_train_mod1.obs.copy()
        )
        ad_train_mean_enc.obsm['X_model'] = ad_train_mean_enc.X.copy()
        ad_mosaic = sc.concat([ad_train_mean_enc, input_test_mod1, input_test_mod2])

        r = eval_mosaic(ad_mosaic, label_key='cell_type', lisi_keys=['mod-batch', 'domain'], 
            use_rep='X_model', use_lisi=True, use_gc=False, use_nmi=False)

        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_model',
            use='nmi', nmi_method='arithmetic')
        print(f'nmi={nmi:.4f}, ari={ari:.4f}')

        ### with harmony
        print('=====================after harmony=====================')
        ad_mosaic_df = pd.DataFrame(ad_mosaic.obsm['X_model'], index=ad_mosaic.obs_names.to_numpy())
        ad_mosaic_df['batch'] = ad_mosaic.obs['mod-batch'].to_numpy()
        ad_mosaic.obsm['X_model_harmony'] = harmony([ad_mosaic_df])[0]

        r = eval_mosaic(ad_mosaic, label_key='cell_type', lisi_keys=['mod-batch', 'domain'], use_rep='X_model_harmony', 
                        use_lisi=True, use_gc=False, use_nmi=False)

        nmi, ari = eval_clustering(
            ad_mosaic, label_key='cell_type', cluster_key='cluster', resolutions=None, use_rep='X_model_harmony',
            use='nmi', nmi_method='arithmetic')
        print(f'nmi={nmi:.4f}, ari={ari:.4f}')



