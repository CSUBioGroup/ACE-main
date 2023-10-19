import argparse
import os
import pickle
import sys
import gc

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
from sklearn.model_selection import train_test_split
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
from matchclot.preprocessing.preprocess import lsiTransformer, harmony, tfidfTransformer
from matchclot.utils.dataloaders import ModalityMatchingDataset
from matchclot.utils.hyperparameters import (
    defaults_common,
    defaults_GEX2ADT,
    defaults_GEX2ATAC,
    baseline_GEX2ATAC,
    baseline_GEX2ADT,
)


# running command
# python sec1_case4.py --DATASETS_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/match_modality/" \
#                 --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot/case4_cite" \
#                 --SAVE_EMBEDDINGS True \
#                 'GEX2ADT' \
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
            "openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_",
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
        input_train_mod1_0 = ad.read_h5ad(par["input_train_mod1"])  # adt
        print("input_train_mod1.shape", input_train_mod1_0.shape)
        input_train_mod2_0 = ad.read_h5ad(par["input_train_mod2"])  # rna
        print("input_train_mod2.shape", input_train_mod2_0.shape)
        sol_train = ad.read_h5ad(par["input_train_sol"])  # ground truth matching
        # Apply the same ordering of mod2 profiles as mod1
        input_train_mod2_0 = input_train_mod2_0[sol_train.to_df().values.argmax(1)]
        input_train_mod2_0.obs_names = input_train_mod1_0.obs_names.to_numpy()

        # Load private test data, used for transductive LSI + Harmony preprocessing
        print("loading private test data")
        input_test_mod1_0 = ad.read_h5ad(par["input_test_mod1"])  # adt
        print("input_test_mod1.shape", input_test_mod1_0.shape)  
        input_test_mod2_0 = ad.read_h5ad(par["input_test_mod2"])  # rna
        print("input_test_mod2.shape", input_test_mod2_0.shape)

        tmp = input_train_mod1_0.copy()
        input_train_mod1_0 = input_train_mod2_0.copy()
        input_train_mod2_0 = tmp.copy()
        tmp = input_test_mod1_0.copy()
        input_test_mod1_0 = input_test_mod2_0.copy()
        input_test_mod2_0 = tmp.copy()
        del tmp
        gc.collect()

        mod1 = input_train_mod1_0.var["feature_types"][0]
        mod2 = input_train_mod2_0.var["feature_types"][0]

        split_trainval_by = 'batch'

    assert mod1 == "GEX"  # mod1 is always GEX, mod2 is either ADT or ATAC

    for p in [0.1, 0.2, 0.4, 0.8]:
        for repeat in range(3):
            new_train_idx = np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/cite/p={p}_r={repeat}_new_train_idx.npy')
            test_rna_idx = np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/cite/p={p}_r={repeat}_test_rna_idx.npy')
            test_adt_idx = np.load(f'/home/yanxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/cite/p={p}_r={repeat}_test_adt_idx.npy')
            
            # cobolt use phase_mod2
            input_train_mod1 = input_train_mod1_0[new_train_idx].copy()
            input_train_mod2 = input_train_mod2_0[new_train_idx].copy()
            input_test_mod1  = input_test_mod1_0[test_rna_idx].copy()
            input_test_mod2  = input_test_mod2_0[test_adt_idx].copy()

            # Load or fit LSI preprocessing
            path = os.path.join(par["output_pretrain"], f'p={p}_repeat={repeat}')

            # Define train and validation split
            fold_number = args.VALID_FOLD
            print("fold_number:", fold_number)
            trial_dump_folder = os.path.join(path, str(fold_number)) # path+'/0'
            if os.path.exists(os.path.join(trial_dump_folder, 'output/test_emb_mod1.pt')):
                continue

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
                        path + "/lsi_GEX_transformer.pickle",
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
                    
            if not os.path.exists(os.path.join(trial_dump_folder, 'model.best.pth')):
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
            else:
                print(f'==> skip training of {p} and {repeat}')

            # ++++++++++++++++++++++++++
            ##### run
            # ++++++++++++++++++++++++++

            # N_cells = input_test_mod1.shape[0]
            # print(
            #     "mod1 cells:", input_test_mod1.shape[0], "mod2 cells:", input_test_mod2.shape[0]
            # )
            # assert input_test_mod2.shape[0] == N_cells

            # Load pretrained models and ensemble predictions
            # sim_matrix = np.zeros((N_cells, N_cells))
            for fold in range(0, 9):
                weight_file = os.path.join(trial_dump_folder, 'model.best.pth') # path + "/" + str(fold) + f"/model.best.pth"
                OUTPUT_PATH = os.path.join(trial_dump_folder, f'output')
                if os.path.exists(weight_file):
                    print("Loading weights from " + weight_file)
                    weight = torch.load(weight_file, map_location="cpu")
                    os.makedirs(OUTPUT_PATH, exist_ok=True)

                    # Define modality encoders
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
                        ).to(device)
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
                        ).to(device)

                    # Load pretrained weights
                    model.load_state_dict(weight)

                    # Load torch datasets
                    dataset_test_rna = ModalityMatchingDataset(
                        pd.DataFrame(gex_private), 
                        pd.DataFrame(np.zeros((gex_private.shape[0], mod2_private.shape[1])))
                    )
                    dataset_test_adt = ModalityMatchingDataset(
                        pd.DataFrame(np.zeros((mod2_private.shape[0], gex_private.shape[1]))), 
                        pd.DataFrame(mod2_private)
                    )
                    dataset_train = ModalityMatchingDataset(
                        pd.DataFrame(gex_all), pd.DataFrame(mod2_all)
                    )
                    data_train = torch.utils.data.DataLoader(
                        dataset_train, 128, shuffle=False, num_workers=4
                    )
                    data_test_rna = torch.utils.data.DataLoader(dataset_test_rna, 32, shuffle=False)
                    data_test_adt = torch.utils.data.DataLoader(dataset_test_adt, 32, shuffle=False)

                    # Predict on test set
                    all_emb_mod1_train, all_emb_mod2_train = [], []
                    all_emb_mod1, all_emb_mod2 = [], []
                    indexes = []
                    model.eval()

                    for batch in data_train:
                        x1 = batch["features_first"].float()
                        x2 = batch["features_second"].float()
                        # The model applies the GEX encoder to the second argument, here x1
                        logits, features_mod2, features_mod1 = model(
                            x2.to(device), x1.to(device)
                        )

                        all_emb_mod1_train.append(features_mod1.detach().cpu())
                        all_emb_mod2_train.append(features_mod2.detach().cpu())
                    all_emb_mod1_train = torch.cat(all_emb_mod1_train)
                    all_emb_mod2_train = torch.cat(all_emb_mod2_train)

                    for batch in data_test_rna:
                        x1 = batch["features_first"].float()
                        x2 = batch["features_second"].float()
                        # The model applies the GEX encoder to the second argument, here x1
                        logits, features_mod2, features_mod1 = model(
                            x2.to(device), x1.to(device)
                        )
                        all_emb_mod1.append(features_mod1.detach().cpu())
                    for batch in data_test_adt:
                        x1 = batch["features_first"].float()
                        x2 = batch["features_second"].float()
                        # The model applies the GEX encoder to the second argument, here x1
                        logits, features_mod2, features_mod1 = model(
                            x2.to(device), x1.to(device)
                        )
                        all_emb_mod2.append(features_mod2.detach().cpu())
                    all_emb_mod1 = torch.cat(all_emb_mod1)
                    all_emb_mod2 = torch.cat(all_emb_mod2)

                    # Save the embeddings concatenated according to the true order and predicted order
                    if args.SAVE_EMBEDDINGS:
                        # Assumes that the two modalities have the cells in the same order
                        # all_emb_mod12_true = torch.cat((all_emb_mod1, all_emb_mod2), dim=1)
                        file1 = os.path.join(OUTPUT_PATH, "test_emb_mod1.pt")
                        file2 = os.path.join(OUTPUT_PATH, 'test_emb_mod2.pt')
                        print("Test embeddings saved")
                        torch.save(all_emb_mod1, file1)
                        torch.save(all_emb_mod2, file2)

                        all_emb_mod12_train = torch.cat((all_emb_mod1_train, all_emb_mod2_train), dim=1) 
                        file = os.path.join(OUTPUT_PATH, "train_emb_mod12.pt")
                        torch.save(all_emb_mod12_train, file)
                        print("Traing embeddings saved to", file)

                    break
                else:
                    raise ValueError("weight file not exist") 
