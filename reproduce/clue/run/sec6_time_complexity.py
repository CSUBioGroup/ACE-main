import gc
import logging
import os
import pickle
import h5py
import datetime

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
from preprocessing import harmony

# data_dir = '/home/cb213/local/cache3/yxh/Data/seurat4-CITE-reference'
data_dir2 = '/home/yanxh/gitrepo/multi-omics-matching/tmp_outputs/time_complx/inputs'
out_dir = '/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/clue/time_complx'

print('Reading `mtx` files...')
_path = '/home/sda1/yanxh/data/seurat-CITE-reference/cite.h5'
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
    
    # adt_norm_data = np.array(f['adt_norm_data'], dtype=np.float32)
    adt_count_data = np.array(f['adt_count_data'], dtype=np.float32)
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

ad_cite.obsm['adt'] = sps.csr_matrix(adt_count_data)

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

######### 
for rate in [0.01, 0.1, 0.2, 0.4, 0.8, 1.0]:
    save_dir = join(out_dir, f'rate_{rate}')
    os.makedirs(save_dir, exist_ok=True)

    smp_names = pd.read_csv(join(data_dir2, f'names_{rate}.csv'))['0'].values
    n_smp = len(smp_names)
    n_interval = n_smp // 3
    ad_cite_subset = ad_cite[smp_names].copy()
    batch = ad_cite_subset.obs.batch.to_numpy()

    input_train_mod1 = sc.AnnData(ad_cite_subset[:n_interval].X.copy(), obs=ad_cite_subset[:n_interval].obs.copy())
    input_train_mod2 = sc.AnnData(ad_cite_subset[:n_interval].obsm['adt'].copy(), obs=ad_cite_subset[:n_interval].obs.copy())
    input_train_mod1.layers['counts'] = input_train_mod1.X.copy()
    input_train_mod2.layers['counts'] = input_train_mod2.X.copy()
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

    gex = input_train_mod1
    other = input_train_mod2

    print('Preprocessing GEX...')
    gex_prep = utils.GEXPreprocessing(n_comps=100, n_genes=n_genes, merge_adt=False)
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
    stage1_st_time = datetime.datetime.now()
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
            {"gex": gex, "other": other},
            max_epochs = 1000
        )
        model.save(os.path.join(save_dir, "pretrain.dill"))
    else:
        pass
    stage1_ed_time = datetime.datetime.now()
    stage1_time_cost = (stage1_ed_time - stage1_st_time).total_seconds()

    # =========================
    ###### running
    # =========================
    input_test_mod1 = sc.AnnData(
        ad_cite_subset[n_interval:int(2*n_interval)].X.copy(), 
        obs=ad_cite_subset[n_interval:int(2*n_interval)].obs.copy()
    )
    input_test_mod2 = sc.AnnData(
        ad_cite_subset[int(2*n_interval):int(3*n_interval)].obsm['adt'].copy(), 
        obs=ad_cite_subset[int(2*n_interval):int(3*n_interval)].obs.copy()
    )
    input_test_mod1.obs["uid"] = [f"test-gex-{i}" for i in range(input_test_mod1.shape[0])]
    input_test_mod2.obs["uid"] = [f"test-adt-{i}" for i in range(input_test_mod2.shape[0])]

    logging.info('Concatenating training and test data...')
    input_mod1 = ad.concat(
        {"train": input_train_mod1, "test": input_test_mod1},
        axis=0, join="outer", merge="same", label="group",
        fill_value=0, index_unique="-"
    )
    input_mod1.uns["feature_type"] = 'gex'
    del input_train_mod1, input_test_mod1
    gc.collect()
    input_mod2 = ad.concat(
        {"train": input_train_mod2, "test": input_test_mod2},
        axis=0, join="outer", merge="same", label="group",
        fill_value=0, index_unique="-"
    )
    input_mod2.uns["feature_type"] = 'adt'
    del input_train_mod2, input_test_mod2
    gc.collect()

    gex, other = input_mod1, input_mod2

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

    stage2_st_time = datetime.datetime.now()
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

    logging.info('Training model...')
    model.fit(
        {"gex": gex, "other": other},
        align_burnin=0, max_epochs=50,  # 50 or 5
        patience=8, reduce_lr_patience=3
    )

    # model.save(os.path.join(save_dir, "train.dill"))

    ###### evaluation
    # from evaluation import eval_mosaic, eval_specific_mod, eval_bridge, print_results, eval_asw, eval_lisi, eval_clustering
#     mod1_enc = model.encode_data('gex' if mod1_feature_type=='GEX' else 'other', input_mod1[input_mod1.obs["group"] == "test"])
#     mod2_enc = model.encode_data('gex' if mod2_feature_type=='GEX' else 'other', input_mod2[input_mod2.obs["group"] == "test"])

#     mod1_train_enc = model.encode_data('gex' if mod1_feature_type=='GEX' else 'other', input_mod1[input_mod1.obs["group"] == "train"])
#     mod2_train_enc = model.encode_data('gex' if mod2_feature_type=='GEX' else 'other', input_mod2[input_mod2.obs["group"] == "train"])

    stage2_ed_time = datetime.datetime.now()
    stage2_time_cost = (stage2_ed_time - stage2_st_time).total_seconds()

    print('=============================')
    print(f'Rate={rate}')
    print('Time cost: ', (stage2_time_cost + stage1_time_cost))
    print('=============================')