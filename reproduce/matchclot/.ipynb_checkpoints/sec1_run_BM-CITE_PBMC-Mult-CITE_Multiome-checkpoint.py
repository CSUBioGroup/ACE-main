import argparse
import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import scipy.sparse as sps
import scipy.io as sio
import scanpy as sc
from os.path import join
from distutils.util import strtobool

from catalyst.utils import set_global_seed

sys.path.insert(0, '/home/yanxh/gitrepo/multi-omics-matching/MatchCLOT-main')
import matchclot
from matchclot.embedding.models import Modality_CLIP, Encoder
from matchclot.matching.matching import OT_matching, MWB_matching
from matchclot.preprocessing.preprocess import harmony
from matchclot.run.evaluate import evaluate
from matchclot.utils.dataloaders import ModalityMatchingDataset
from matchclot.utils.hyperparameters import (
    defaults_common,
    defaults_GEX2ADT,
    defaults_GEX2ATAC,
    baseline_GEX2ATAC,
    baseline_GEX2ADT,
)

# inference commands

# case2: CITE
# python sec1_run_BM-CITE_PBMC-Mult-CITE_Multiome.py --DATASETS_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/match_modality/" \
#               --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#               --SAVE_EMBEDDINGS True \
#               --ckpt 'last' \
#                 'GEX2ADT' 

# case2: Multiome
# python sec1_run_BM-CITE_PBMC-Mult-CITE_Multiome.py --DATASETS_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/datasets/match_modality/" \
#               --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#               --SAVE_EMBEDDINGS True \
#               --ckpt 'last' \
#                 'GEX2ATAC' 

# case1: BM-CITE
# python sec1_run_BM-CITE_PBMC-Mult-CITE_Multiome.py --DATASETS_PATH "/home/yanxh/data/Seurat_demo_data/bm_cite/" \
#               --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#               --SAVE_EMBEDDINGS True \
#               --HARMONY False \
#               --ckpt 'best' \
#               'BM-CITE' \
#               --N_LSI_COMPONENTS_ADT 25

# case1: PBMC-Mult
# python sec1_run_BM-CITE_PBMC-Mult-CITE_Multiome.py --DATASETS_PATH "/home/sda1/yanxh/data/Seurat_demo_data/pbmc_multiome" \
#               --PRETRAIN_PATH "/home/yanxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/matchclot" \
#               --SAVE_EMBEDDINGS True \
#               --HARMONY False \
#               --ckpt 'best' \
#               'PBMC-Mult' \


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

    parser.add_argument(
            "--ckpt",
            default='best',
            type=str,
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
        "input_pretrain": pretrain_path,
    }

    # Overwrite configurations for ablation study
    if args.HYPERPARAMS is False:
        if is_multiome:
            for hyperparam, baseline_value in baseline_GEX2ATAC.items():
                setattr(args, hyperparam, baseline_value)
        else:
            for hyperparam, baseline_value in baseline_GEX2ADT.items():
                setattr(args, hyperparam, baseline_value)
    print("args:", args, "unknown_args:", unknown_args)

    # Load data
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

        mod1 = 'GEX'
        mod2 = 'ADT'

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

        mod1 = 'GEX'
        mod2 = 'ATAC'

    N_cells = input_test_mod1.shape[0]
    print(
        "mod1 cells:", input_test_mod1.shape[0], "mod2 cells:", input_test_mod2.shape[0]
    )
    assert input_test_mod2.shape[0] == N_cells

    # Load and apply LSI transformation
    with open(par["input_pretrain"] + "/lsi_GEX_transformer.pickle", "rb") as f:
        try:
            lsi_transformer_gex = pickle.load(f)
        except ModuleNotFoundError:
            sys.modules["resources"] = matchclot
    if is_multiome:
        with open(par["input_pretrain"] + "/lsi_ATAC_transformer.pickle", "rb") as f:
            try:
                lsi_transformer_atac = pickle.load(f)
            except ModuleNotFoundError:
                sys.modules["resources"] = matchclot
        if args.TRANSDUCTIVE:
            gex_train = lsi_transformer_gex.transform(input_train_mod1)
            mod2_train = lsi_transformer_atac.transform(input_train_mod2)
        gex_test = lsi_transformer_gex.transform(input_test_mod1)
        mod2_test = lsi_transformer_atac.transform(input_test_mod2)
    else:
        if args.TRANSDUCTIVE:
            gex_train = lsi_transformer_gex.transform(input_train_mod1)
            mod2_train = input_train_mod2.to_df()
        gex_test = lsi_transformer_gex.transform(input_test_mod1)
        mod2_test = input_test_mod2.to_df()

    if args.HARMONY:
        # Apply Harmony batch effect correction
        gex_test["batch"] = input_test_mod1.obs.batch
        mod2_test["batch"] = input_test_mod2.obs.batch

        if args.TRANSDUCTIVE:
            # Transductive setting
            gex_train["batch"] = input_train_mod1.obs.batch
            (
                gex_train,
                gex_test,
            ) = harmony([gex_train, gex_test])
            mod2_train["batch"] = input_train_mod2.obs.batch
            (
                mod2_train,
                mod2_test,
            ) = harmony([mod2_train, mod2_test])
        else:
            (gex_test,) = harmony([gex_test])
            (mod2_test,) = harmony([mod2_test])
    else:
        gex_train = gex_train.values
        gex_test  = gex_test.values
        mod2_train = mod2_train.values
        mod2_test  = mod2_test.values

    # Load pretrained models and ensemble predictions
    sim_matrix = np.zeros((N_cells, N_cells))
    for fold in range(0, 9):
        weight_file = par["input_pretrain"] + "/" + str(fold) + f"/model.{args.ckpt}.pth"
        OUTPUT_PATH = os.path.join(par["input_pretrain"], str(fold), f'output_{args.ckpt}')
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
            dataset_test = ModalityMatchingDataset(
                pd.DataFrame(gex_test), pd.DataFrame(mod2_test)
            )
            dataset_train = ModalityMatchingDataset(
                pd.DataFrame(gex_train), pd.DataFrame(mod2_train)
            )
            data_train = torch.utils.data.DataLoader(
                dataset_train, 128, shuffle=False, num_workers=4
            )
            data_test = torch.utils.data.DataLoader(dataset_test, 32, shuffle=False)

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

            for batch in data_test:
                x1 = batch["features_first"].float()
                x2 = batch["features_second"].float()
                # The model applies the GEX encoder to the second argument, here x1
                logits, features_mod2, features_mod1 = model(
                    x2.to(device), x1.to(device)
                )

                all_emb_mod1.append(features_mod1.detach().cpu())
                all_emb_mod2.append(features_mod2.detach().cpu())
            all_emb_mod1 = torch.cat(all_emb_mod1)
            all_emb_mod2 = torch.cat(all_emb_mod2)

            # Save the embeddings concatenated according to the true order and predicted order
            if args.SAVE_EMBEDDINGS:
                # Assumes that the two modalities have the cells in the same order
                all_emb_mod12_true = torch.cat((all_emb_mod1, all_emb_mod2), dim=1)
                file = os.path.join(OUTPUT_PATH, "test_emb_mod12.pt")
                torch.save(all_emb_mod12_true, file)
                print("Test embeddings saved to", file)

                all_emb_mod12_train = torch.cat((all_emb_mod1_train, all_emb_mod2_train), dim=1) 
                file = os.path.join(OUTPUT_PATH, "train_emb_mod12.pt")
                torch.save(all_emb_mod12_train, file)
                print("Traing embeddings saved to", file)

            # Calculate the cosine similarity matrix and add it to the ensemble
            # sim_matrix += (all_emb_mod1 @ all_emb_mod2.T).detach().cpu().numpy()
            break

    # save the full similarity matrix
    # np.save("similarity_matrix.npy", sim_matrix)

    if args.BATCH_LABEL_MATCHING:
        # Split matching by batch label
        mod1_splits = set(input_test_mod1.obs["batch"])
        mod2_splits = set(input_test_mod2.obs["batch"])
        splits = mod1_splits | mod2_splits
        matching_matrices, mod1_obs_names, mod2_obs_names = [], [], []
        mod1_obs_index = input_test_mod1.obs.index
        mod2_obs_index = input_test_mod2.obs.index
        print("batch label splits", splits)

        for split in splits:
            print("matching split", split)
            mod1_msk = input_test_mod1.obs["batch"] == split
            mod2_msk = input_test_mod2.obs["batch"] == split
            mod1_obs_names.append(input_test_mod1[mod1_msk].obs_names.to_numpy())
            mod2_obs_names.append(input_test_mod2[mod2_msk].obs_names.to_numpy())

            sim_matrix_split = (all_emb_mod1[mod1_msk] @ all_emb_mod2[mod2_msk].T).detach().cpu().numpy()
            # mod1_indexes = mod1_obs_index.get_indexer(mod1_split.obs_names)
            # mod2_indexes = mod2_obs_index.get_indexer(mod2_split.obs_names)
            # sim_matrix_split = sim_matrix[np.ix_(mod1_indexes, mod2_indexes)]
            # save the split similarity matrix
            # np.save("test_similarity_matrix"+str(split)+".npy", sim_matrix_split)

            if args.OT_MATCHING:
                # Compute OT matching
                matching_matrices.append(
                    OT_matching(sim_matrix_split, entropy_reg=args.OT_ENTROPY)
                )
            else:
                # Max-weight bipartite matching
                matching_matrices.append(MWB_matching(sim_matrix_split))

        # Assemble the matching matrices and reorder according to the original order of cell profiles
        matching_matrix = scipy.sparse.block_diag(matching_matrices, format="csc")
        mod1_obs_names = pd.Index(np.concatenate(mod1_obs_names))
        mod2_obs_names = pd.Index(np.concatenate(mod2_obs_names))
        matching_matrix = matching_matrix[
            mod1_obs_names.get_indexer(mod1_obs_index), :
        ][:, mod2_obs_names.get_indexer(mod2_obs_index)]
    else:
        if args.OT_MATCHING:
            # Compute OT matching
            matching_matrix = OT_matching(sim_matrix, entropy_reg=args.OT_ENTROPY)
        else:
            # Max-weight bipartite matching
            matching_matrix = MWB_matching(sim_matrix)
        matching_matrix = pd.DataFrame(matching_matrix)

    out = ad.AnnData(
        X=matching_matrix,
        uns={
            # "dataset_id": input_test_mod1.uns["dataset_id"],
            "method_id": "MatchCLOT",
        },
    )

    # Save the matching matrix
    out.write_h5ad(os.path.join(OUTPUT_PATH, "predict_match.h5ad"), compression="gzip")
    print("Prediction saved to", os.path.join(OUTPUT_PATH, "predict_match.h5ad"))

    # Load the solution for evaluation
    if args.TASK in ['GEX2ADT', 'GEX2ATAC']:
        if is_multiome:
            sol_path = os.path.join(
                args.DATASETS_PATH,
                "openproblems_bmmc_multiome_phase2_rna"
                "/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_",
            )
        else:
            sol_path = os.path.join(
                args.DATASETS_PATH,
                "openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna"
                ".censor_dataset.output_",
            )

        sol = ad.read_h5ad(sol_path + "test_sol.h5ad")
    else:
        print("For evaluation assuming cell order is the same in the two modalities")
        # sol.X.toarray() will be called
        sol = ad.AnnData(X=scipy.sparse.eye(matching_matrix.shape[0]))

    # Score the prediction and save the results
    scores_path = os.path.join(OUTPUT_PATH, "scores.txt")
    evaluate(out, sol, scores_path)


if __name__ == "__main__":
    main()
