import argparse
import os
import pickle
import sys

import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sps
import scanpy as sc
import scipy.io as sio
import torch
from catalyst import dl
from os.path import join
from catalyst.utils import set_global_seed
from sklearn.model_selection import LeaveOneGroupOut
from distutils.util import strtobool
from catalyst.engines.torch import (
        CPUEngine,
        DataParallelEngine,
        DistributedDataParallelEngine,
        GPUEngine,
    )

sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/MatchCLOT-main')
from matchclot.embedding.catalyst_tools import scRNARunner, CustomMetric
from matchclot.embedding.models import Modality_CLIP, Encoder
from matchclot.preprocessing.preprocess import lsiTransformer, harmony
from matchclot.utils.dataloaders import ModalityMatchingDataset
from matchclot.utils.hyperparameters import (
    defaults_common,
    defaults_GEX2ADT,
    defaults_GEX2ATAC,
    baseline_GEX2ATAC,
    baseline_GEX2ADT,
)

###  training commands !!!!!

# case2: CITE
# python sec1_train_BM-CITE_PBMC-Mult_CITE_Multiome.py --DATASETS_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/match_modality/" \
#                 --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#                 'GEX2ADT' \
#                 --N_EPOCHS 1000 

# case2: Multiome
# python sec1_train_BM-CITE_PBMC-Mult_CITE_Multiome.py --DATASETS_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/match_modality/" \
#                 --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#                 'GEX2ATAC' \
#                 --N_EPOCHS 1000 

# case1: BM-CITE
# python sec1_train_BM-CITE_PBMC-Mult_CITE_Multiome.py --DATASETS_PATH "/home/yanxh/data/Seurat_demo_data/bm_cite/" \
#                 --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#                 --HARMONY False \
#                 'BM-CITE' \
#                 --N_EPOCHS 1000 \
#                 --N_LSI_COMPONENTS_ADT 25  

# case1: PBMC-Mult
# python sec1_train_BM-CITE_PBMC-Mult_CITE_Multiome.py --DATASETS_PATH "/home/sda1/yanxh/data/Seurat_demo_data/pbmc_multiome" \
#                 --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#                 --HARMONY False \
#                 'PBMC-Mult' \
#                 --N_EPOCHS 1000 



if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    # setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define argument parsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="TASK")

    # Common args
    for key, value in defaults_common.items():
        parser.add_argument(
            "--" + key,
            default=value,
            type=(lambda x: bool(strtobool(x))) if type(value) == bool else type(value),
        )

    # GEX2ADT args
    parser_GEX2ADT = subparsers.add_parser("GEX2ADT", help="train GEX2ADT model")
    for key, value in defaults_GEX2ADT.items():
        parser_GEX2ADT.add_argument("--" + key, default=value, type=type(value))

    # BM-CITE follows GEX2ADT args
    parser_BM_CITE = subparsers.add_parser("BM-CITE", help="train GEX2ADT model")
    for key, value in defaults_GEX2ADT.items():
        parser_BM_CITE.add_argument("--" + key, default=value, type=type(value))

    # GEX2ATAC args
    parser_GEX2ATAC = subparsers.add_parser("GEX2ATAC", help="train GEX2ATAC model")
    for key, value in defaults_GEX2ATAC.items():
        parser_GEX2ATAC.add_argument("--" + key, default=value, type=type(value))
        
    # PBMC-Mult args
    parser_PBMC_Mult = subparsers.add_parser("PBMC-Mult", help="train GEX2ATAC model")
    for key, value in defaults_GEX2ATAC.items():
        parser_PBMC_Mult.add_argument("--" + key, default=value, type=type(value))

    # Parse args
    args, unknown_args = parser.parse_known_args()

    # Set global random seed
    set_global_seed(args.SEED)

    # Date in format YYMMDDHHMMSS
    date = "".join(
        [
            c if c.isnumeric() else ""
            for c in str(pd.Timestamp("today").to_pydatetime())
        ][2:19]
    )

    # Define file paths
    if args.TASK == "GEX2ADT":
        dataset_path = os.path.join(
            args.DATASETS_PATH,
            "openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_",
        )
        pretrain_path = os.path.join(
            args.PRETRAIN_PATH, "GEX2ADT"
        )  # Path for saving the trained model
        is_multiome = False
    elif args.TASK == "GEX2ATAC":
        dataset_path = os.path.join(
            args.DATASETS_PATH,
            "openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_",
        )
        pretrain_path = os.path.join(args.PRETRAIN_PATH, "GEX2ATAC")
        is_multiome = True
    elif args.TASK == 'BM-CITE':
        dataset_path = args.DATASETS_PATH
        pretrain_path = os.path.join(
            args.PRETRAIN_PATH, 'BM-CITE'
        )
        is_multiome = False
    elif args.TASK == 'PBMC-Mult':
        dataset_path = args.DATASETS_PATH
        pretrain_path = os.path.join(
            args.PRETRAIN_PATH, 'PBMC-Mult'
        )
        is_multiome = True
    else:
        raise ValueError("Unknown task: " + args.TASK)

    par = {
        "input_train_mod1": f"{dataset_path}train_mod1.h5ad",
        "input_train_mod2": f"{dataset_path}train_mod2.h5ad",
        "input_train_sol": f"{dataset_path}train_sol.h5ad",
        "input_test_mod1": f"{dataset_path}test_mod1.h5ad",
        "input_test_mod2": f"{dataset_path}test_mod2.h5ad",
        "output_pretrain": pretrain_path,
        "input_pretrain": pretrain_path,
    }
    os.makedirs(par["output_pretrain"], exist_ok=True)

    # Overwrite configurations for ablation study
    if args.HYPERPARAMS is False:
        if is_multiome:
            for hyperparam, baseline_value in baseline_GEX2ATAC.items():
                setattr(args, hyperparam, baseline_value)
        else:
            for hyperparam, baseline_value in baseline_GEX2ADT.items():
                setattr(args, hyperparam, baseline_value)
    print("args:", args, "unknown_args:", unknown_args)

    if args.TASK in ['GEX2ADT', 'GEX2ATAC']:
        # Load train data
        print("loading train data")
        input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
        print("input_train_mod1.shape", input_train_mod1.shape)
        input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
        print("input_train_mod2.shape", input_train_mod2.shape)
        sol_train = ad.read_h5ad(par["input_train_sol"])  # ground truth matching
        # Apply the same ordering of mod2 profiles as mod1
        input_train_mod2 = input_train_mod2[sol_train.to_df().values.argmax(1)]

        # Load private test data, used for transductive LSI + Harmony preprocessing
        print("loading private test data")
        input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])
        print("input_test_mod1.shape", input_test_mod1.shape)
        input_test_mod2 = ad.read_h5ad(par["input_test_mod2"])
        print("input_test_mod2.shape", input_test_mod2.shape)

        mod1 = input_train_mod1.var["feature_types"][0]
        mod2 = input_train_mod2.var["feature_types"][0]

        split_trainval_by = 'batch'

    elif args.TASK == 'BM-CITE':
        print('Reading `mtx` files...')
        rna_count_mat = sps.csr_matrix(sio.mmread(join(args.DATASETS_PATH, 'rna_mat_count.mtx')).T)
        adt_count_mat = sps.csr_matrix(sio.mmread(join(args.DATASETS_PATH, 'adt_mat_count.mtx')).T)
        rna_names = pd.read_csv(join(args.DATASETS_PATH, 'gene_names.csv'))['x'].to_numpy()
        adt_names = pd.read_csv(join(args.DATASETS_PATH, 'adt_names.csv'))['x'].to_numpy()
        cell_names = pd.read_csv(join(args.DATASETS_PATH, 'cell_names.csv'))['x'].to_numpy()
        meta_data = pd.read_csv(join(args.DATASETS_PATH, 'metadata.csv'), index_col=0)
        meta_data['batch'] = meta_data['donor'].to_numpy()
        train_idx = np.where((meta_data.batch=='batch1').to_numpy())[0]
        test_idx  = np.where((meta_data.batch=='batch2').to_numpy())[0]

        print('Reading `h5ad` files...')
        input_train_mod1 = sc.AnnData(sps.csr_matrix(rna_count_mat[train_idx]), obs=meta_data.iloc[train_idx])
        input_train_mod2 = sc.AnnData(sps.csr_matrix(adt_count_mat[train_idx]), obs=meta_data.iloc[train_idx])
        input_test_mod1 = sc.AnnData(sps.csr_matrix(rna_count_mat[test_idx]), obs=meta_data.iloc[test_idx])
        input_test_mod2 = sc.AnnData(sps.csr_matrix(adt_count_mat[test_idx]), obs=meta_data.iloc[test_idx])

        # set var names
        input_train_mod1.var_names = input_test_mod1.var_names = rna_names
        input_train_mod2.var_names = input_test_mod2.var_names = adt_names

        # split train val index
        n_train = input_train_mod1.shape[0]
        trainval = np.zeros(n_train)
        # val index
        trainval_idx1 = np.random.choice(np.arange(n_train), int(0.2*n_train), replace=False)
        trainval[trainval_idx1] = 1  # val idx=1
        trainval = ['val' if _ else 'train' for _ in trainval] 
        input_train_mod1.obs['trainval'] = input_train_mod2.obs['trainval'] = trainval

        mod1 = 'GEX'
        mod2 = 'ADT'

        split_trainval_by = 'trainval'
    
    elif args.TASK == 'PBMC-Mult':
        print('Reading `mtx` files...')
        rna_count_mat = sps.csr_matrix(sio.mmread(join(args.DATASETS_PATH, 'rna_mat_count.mtx')).T)
        atac_count_mat = sps.csr_matrix(sio.mmread(join(args.DATASETS_PATH, 'atac_mat_count.mtx')).T)
        rna_names = pd.read_csv(join(args.DATASETS_PATH, 'gene_names.csv'))['x'].to_numpy()
        atac_names = pd.read_csv(join(args.DATASETS_PATH, 'atac_names.csv'))['x'].to_numpy()
        cell_names = pd.read_csv(join(args.DATASETS_PATH, 'cell_names.csv'))['x'].to_numpy()
        meta_data = pd.read_csv(join(args.DATASETS_PATH, 'metadata.csv'), index_col=0)
        train_idx = pd.read_csv(join(args.DATASETS_PATH, 'train_idx.csv'))['0'].to_numpy()
        test_idx  = pd.read_csv(join(args.DATASETS_PATH, 'test_idx.csv'))['0'].to_numpy()

        print('Reading `h5ad` files...')
        input_train_mod1 = sc.AnnData(sps.csr_matrix(rna_count_mat[train_idx]), obs=meta_data.loc[cell_names[train_idx]])
        input_train_mod2 = sc.AnnData(sps.csr_matrix(atac_count_mat[train_idx]), obs=meta_data.loc[cell_names[train_idx]])
        input_test_mod1 = sc.AnnData(sps.csr_matrix(rna_count_mat[test_idx]), obs=meta_data.loc[cell_names[test_idx]])
        input_test_mod2 = sc.AnnData(sps.csr_matrix(atac_count_mat[test_idx]), obs=meta_data.loc[cell_names[test_idx]])

        # set var names
        input_train_mod1.var_names = input_test_mod1.var_names = rna_names
        input_train_mod2.var_names = input_test_mod2.var_names = atac_names

        # set batch 
        input_train_mod1.obs['batch'] = 'batch1'
        input_train_mod2.obs['batch'] = 'batch1'
        input_test_mod1.obs['batch'] = 'batch1'
        input_test_mod2.obs['batch'] = 'batch1'

        # split train val index
        n_train = input_train_mod1.shape[0]
        trainval = np.zeros(n_train)
        # val index
        trainval_idx1 = np.random.choice(np.arange(n_train), int(0.2*n_train), replace=False)
        trainval[trainval_idx1] = 1  # val idx=1
        trainval = ['val' if _ else 'train' for _ in trainval] 
        input_train_mod1.obs['trainval'] = input_train_mod2.obs['trainval'] = trainval

        mod1 = 'GEX'
        mod2 = 'ATAC'

        split_trainval_by = 'trainval'
        
    assert mod1 == "GEX"  # mod1 is always GEX, mod2 is either ADT or ATAC

    # Define train and validation split
    fold_number = args.VALID_FOLD
    print("fold_number:", fold_number)
    trial_dump_folder = os.path.join(par["output_pretrain"], str(fold_number))
    logo = LeaveOneGroupOut()
    groups = input_train_mod2.obs[split_trainval_by].to_numpy()
    print("GROUPS:", np.unique(groups))
    logo.get_n_splits(input_train_mod2, groups=groups)
    all_splits = list(logo.split(input_train_mod2, groups=groups))
    train_indexes, test_indexes = all_splits[fold_number]
    if len(train_indexes) < len(test_indexes):   # swap two sets
        tmp = test_indexes
        test_indexes = train_indexes
        train_indexes = tmp
    print("len train:", len(train_indexes), "len test:", len(test_indexes))
    print('Train groups:', np.unique(groups[train_indexes]))
    print('Test groups:', np.unique(groups[test_indexes]))

    # Load or fit LSI preprocessing
    path = par["output_pretrain"]

    if (
        os.path.exists(path + "/lsi_GEX_transformer.pickle")
        and args.TRANSDUCTIVE
        and not is_multiome
    ):
        # Avoid re-computing LSI transformation when using cross-validation and transductive LSI
        print("loading lsi transformer from", path)
        # LSI is applied only on GEX and ATAC, not on ADT
        with open(path + "/lsi_GEX_transformer.pickle", "rb") as f:
            lsi_transformer_gex = pickle.load(f)
    elif (
        os.path.exists(path + "/lsi_GEX_transformer.pickle")
        and os.path.exists(path + "/lsi_ATAC_transformer.pickle")
        and args.TRANSDUCTIVE
        and is_multiome
    ):
        with open(path + "/lsi_GEX_transformer.pickle", "rb") as f:
            lsi_transformer_gex = pickle.load(f)
        with open(path + "/lsi_ATAC_transformer.pickle", "rb") as f:
            lsi_transformer_atac = pickle.load(f)
    else:
        print("No lsi transformer found in", path, "creating new one")
        os.makedirs(path, exist_ok=True)

        # Fit GEX LSI
        lsi_transformer_gex = lsiTransformer(
            n_components=args.N_LSI_COMPONENTS_GEX, drop_first=True
        )
        if args.TRANSDUCTIVE:
            print("concatenating gex train and test")
            concatenated_gex = ad.concat(
                [input_train_mod1, input_test_mod1], join="outer"
            )
            print("done, concatenated_gex.shape", concatenated_gex.shape)
            lsi_transformer_gex.fit(concatenated_gex)
            # Save LSI transformation
            with open(path + "/lsi_GEX_transformer.pickle", "wb") as f:
                pickle.dump(lsi_transformer_gex, f)
            print("saved lsi pickle in ", path + "/lsi_GEX_transformer.pickle")
        else:
            lsi_transformer_gex.fit(input_train_mod1)
            with open(path + "/lsi_GEX_transformer.pickle", "wb") as f:
                pickle.dump(lsi_transformer_gex, f)
            print(
                "saved lsi pickle in ",
                trial_dump_folder + "/lsi_GEX_transformer.pickle",
            )

        # LSI is applied only on GEX and ATAC, not on ADT
        if is_multiome:
            # Fit ATAC LSI
            lsi_transformer_atac = lsiTransformer(
                n_components=args.N_LSI_COMPONENTS_ATAC, drop_first=True
            )
            if args.TRANSDUCTIVE:
                print("concatenating atac train and test")
                concatenated_atac = ad.concat(
                    [input_train_mod2, input_test_mod2], join="outer"
                )
                print("done, concatenated_atac.shape", concatenated_atac.shape)
                lsi_transformer_atac.fit(concatenated_atac)
            else:
                lsi_transformer_atac.fit(input_train_mod2)

            # Save LSI transformation
            with open(path + "/lsi_ATAC_transformer.pickle", "wb") as f:
                pickle.dump(lsi_transformer_atac, f)
            print("saved lsi pickle in ", path + "/lsi_ATAC_transformer.pickle")

    # lsi transform
    gex_all = lsi_transformer_gex.transform(input_train_mod1)
    gex_private = lsi_transformer_gex.transform(input_test_mod1)

    if is_multiome:
        mod2_all = lsi_transformer_atac.transform(input_train_mod2)
        mod2_private = lsi_transformer_atac.transform(input_test_mod2)
    else:
        mod2_all = input_train_mod2.to_df()
        mod2_private = input_test_mod2.to_df()

    # Apply Harmony batch effect correction
    if args.HARMONY:
        print('doing harmony')
        gex_all["batch"] = input_train_mod1.obs.batch.to_numpy()
        gex_private["batch"] = input_test_mod1.obs.batch.to_numpy()
        mod2_all["batch"] = input_train_mod2.obs.batch.to_numpy()
        mod2_private["batch"] = input_test_mod2.obs.batch.to_numpy()

        if args.TRANSDUCTIVE:
            # Transductive setting
            gex_all, gex_private = harmony(
                [gex_all, gex_private]
            )
            mod2_all, mod2_private = harmony(
                [mod2_all, mod2_private]
            )
        else:                                # 不管是否transductive，train和valid dataset是同时处理的;
            gex_all, gex_private = harmony(
                [gex_all, gex_private]
            )
            mod2_all, mod2_private = harmony(
                [mod2_all, mod2_private]
            )
    else:
        print('skipping harmony')
        gex_all = gex_all.values
        gex_private = gex_private.values
        mod2_all = mod2_all.values
        mod2_private = mod2_private.values
            
    # np.save(os.path.join(path, 'gex_train_pp.npy'), gex_all)
    # np.save(os.path.join(path, 'other_train_pp.npy'), mod2_all)
    # np.save(os.path.join(path, 'gex_test_pp.npy'), gex_private)
    # np.save(os.path.join(path, 'other_test_pp.npy'), mod2_private)

    # subsetting
    gex_train = gex_all[train_indexes].copy()
    gex_test = gex_all[test_indexes].copy()

    mod2_train = mod2_all[train_indexes].copy()
    mod2_test = mod2_all[test_indexes].copy()

    # Load torch dataloaders
    dataset_train = ModalityMatchingDataset(
        pd.DataFrame(mod2_train), pd.DataFrame(gex_train)
    )
    dataset_test = ModalityMatchingDataset(
        pd.DataFrame(mod2_test), pd.DataFrame(gex_test)
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, args.BATCH_SIZE, shuffle=True, num_workers=4
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, 256, shuffle=False, num_workers=4
    )
    print("loaded dataloaders")

    # Define modality encoders and trainer
    if is_multiome:
        model = Modality_CLIP(
            Encoder=Encoder,
            layers_dims=(
                [args.LAYERS_DIM_ATAC],
                [args.LAYERS_DIM_GEX0, args.LAYERS_DIM_GEX1],
            ),
            dropout_rates=(
                [args.DROPOUT_RATES_ATAC],
                [args.DROPOUT_RATES_GEX0, args.DROPOUT_RATES_GEX1],
            ),
            dim_mod1=args.N_LSI_COMPONENTS_ATAC,
            dim_mod2=args.N_LSI_COMPONENTS_GEX,
            output_dim=args.EMBEDDING_DIM,
            T=args.LOG_T,
            noise_amount=args.SFA_NOISE,
        )
    else:
        model = Modality_CLIP(
            Encoder=Encoder,
            layers_dims=(
                [args.LAYERS_DIM_ADT0, args.LAYERS_DIM_ADT1],
                [args.LAYERS_DIM_GEX0, args.LAYERS_DIM_GEX1],
            ),
            dropout_rates=(
                [args.DROPOUT_RATES_ADT0, args.DROPOUT_RATES_ADT1],
                [args.DROPOUT_RATES_GEX0, args.DROPOUT_RATES_GEX1],
            ),
            dim_mod1=args.N_LSI_COMPONENTS_ADT,
            dim_mod2=args.N_LSI_COMPONENTS_GEX,
            output_dim=args.EMBEDDING_DIM,
            T=args.LOG_T,
            noise_amount=args.SFA_NOISE,
        )

    optimizer = torch.optim.Adam(
        model.parameters(), args.LR, weight_decay=args.WEIGHT_DECAY
    )
    loaders = {
        "train": dataloader_train,
        "valid": dataloader_test,
    }
    runner = scRNARunner()

    # Train model
    runner.train(
        model=model,
        optimizer=optimizer,
        engine=GPUEngine(),   # 只tm能用GPUEngine，其他的有tm的傻逼bug
        loaders=loaders,
        num_epochs=args.N_EPOCHS,
        callbacks=[
            dl.OptimizerCallback(metric_key="loss"),
            dl.CheckpointCallback(
                logdir=trial_dump_folder,
                loader_key="valid",
                metric_key="avg_acc",
                minimize=False,
                topk=1,
            ),
            dl.EarlyStoppingCallback(
                patience=150,
                loader_key="valid",
                metric_key="avg_acc",
                minimize=False,
                min_delta=1e-5,
            ),
            dl.ControlFlowCallbackWrapper(
                base_callback=dl.LoaderMetricCallback(
                    metric=CustomMetric(),
                    input_key=["embeddings_first", "embeddings_second", "temperature"],
                    target_key=["embeddings_second"],
                ),
                ignore_loaders="train",  # Compute metrics only for validation, takes a long time on the training set
            ),
        ],
        verbose=True,
    )
