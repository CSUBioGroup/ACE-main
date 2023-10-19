import os
import pickle
import torch
import gc
import itertools 
from scipy.sparse.csgraph import connected_components

import torch.nn as nn
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import scipy.io as sio
import sys
import math
import scipy.sparse as sps
from os.path import join
from collections import defaultdict
from sklearn.preprocessing import normalize
from annoy import AnnoyIndex

import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.nn.functional as F

def _get_input(modBatch_dict, useReps_dict):
    mod_inputs, mod_inputSizes = {}, {}
    for mod, adas in modBatch_dict.items():
        use_rep = useReps_dict[mod]
        mod_inputs[mod], mod_inputSizes[mod] = [], None

        for ada in adas:
            if ada is None:
                mod_inputs[mod].append(None)
            elif use_rep == 'X':
                mod_inputs[mod].append(sps.csr_matrix(ada.X))
                mod_inputSizes[mod] = ada.X.shape[1]
            else:
                mod_inputs[mod].append(ada.obsm[use_rep])
                mod_inputSizes[mod] = ada.obsm[use_rep].shape[1]

        if mod_inputSizes[mod] is None:
            raise ValueError(f'{mod} is empty')
    return mod_inputs, mod_inputSizes

def check_batch_empty(modBatch_dict, verbose=True):
    mod_names = list(modBatch_dict.keys())
    n_batches = len(modBatch_dict[mod_names[0]])
    batch_contained_mod_ids = []
    for bi in range(n_batches):
        modIds_in_bi = []
        for mi, mod in enumerate(mod_names):
            if modBatch_dict[mod][bi] is not None:
                modIds_in_bi.append(mi)
        if len(modIds_in_bi) == 0:
            raise ValueError(f'batch {bi} empty')

        batch_contained_mod_ids.append(modIds_in_bi)
        if verbose:
            print(f'batch{bi}: {[mod_names[_] for _ in modIds_in_bi]}')
    return batch_contained_mod_ids

def save_model(model, mod_names, ckpt_id, log_dir, stage_label='stage1'):
    os.makedirs(join(log_dir, stage_label), exist_ok=True)
    state = {}
    for mod in mod_names:
        state[mod] = model[mod].state_dict()
    state.update({"epoch":ckpt_id})
    torch.save(state, join(log_dir, f'{stage_label}/model_{ckpt_id:04d}.pth'))

def load_model(model, mod_names, log_dir, ckpt_id=None, stage_label='stage2'):
    ckpt = torch.load(os.path.join(log_dir, f'{stage_label}/model_{ckpt_id:04d}.pth'))
    epoch = ckpt['epoch']
    print(f'loading ckpt at epoch {epoch}')
    for mod in mod_names:
        if mod in ckpt:
            model[mod].load_state_dict(ckpt[mod])
        else:
            print(f'{mod} not found in checkpoint, ignore it')

def stage1_fit_perStep(model, opt, crit, datasets, dls, T):
    step, EPOCH_TOTAL_LOSS, EPOCH_BATCH_LOSSES = 0, 0, []
    for dli, _dl in enumerate(dls):
        tmp_loss = 0
        valid_mod_names = datasets[dli].valid_mod_names
        for iids, batch_data in _dl:
            iids = iids.long().cuda()
            batch_data = [_.cuda() for _ in batch_data]  # list: [batch_mod1, batch_mod2]
            batch_dict = dict(zip(valid_mod_names, batch_data))  # {'rna': batch_rna, 'adt': batch_adt, ...}

            opt.zero_grad()
            feat_dict = model(batch_dict)
            feat_list = [v for k,v in feat_dict.items() if v is not None]
            feats = torch.cat(feat_list, dim=0)
            iids = iids.repeat(len(feat_list))
            step_loss = crit[len(feat_list)]((feats @ feats.T)*T.exp(), iids)
                
            step_loss.backward()
            opt.step()
            step += 1

            tmp_loss += step_loss.item()
            EPOCH_TOTAL_LOSS += step_loss.item()

        EPOCH_BATCH_LOSSES.append(tmp_loss/max(1, len(_dl)))

    return step, EPOCH_TOTAL_LOSS, EPOCH_BATCH_LOSSES

def stage2_fit_perStep(mod_names, model, opt, crit, datasets, dls, T):
    MOD_EPOCH_LOSS = {k:0 for k in mod_names}
    for k in mod_names:
        step = 0
        for dl in dls[k]:
            for batch_data in dl:
                batch_data = batch_data.float().cuda()

                opt[k].zero_grad()
                batch_feat = F.normalize(model[k](batch_data), dim=1, p=2)
                batch_sim = T.exp() * (batch_feat @ batch_feat.T)
                target = torch.arange(batch_sim.shape[0]).cuda()
                batch_loss = crit(batch_sim, target)
                batch_loss.backward()
                opt[k].step()

                MOD_EPOCH_LOSS[k] += batch_loss.item()
                step += 1

        MOD_EPOCH_LOSS[k] /= max(1, step)  # in case empty training batch

    return MOD_EPOCH_LOSS


# ds1 query
# ds2 reference
def nn_approx(ds1, ds2, norm=True, knn=10, metric='manhattan', n_trees=10, include_distances=False):
    if norm:
        ds1 = normalize(ds1)
        ds2 = normalize(ds2) 

    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind, dist = [], []
    for i in range(ds1.shape[0]):
        i_ind, i_dist = a.get_nns_by_vector(ds1[i, :], knn, search_k=-1, include_distances=True)
        ind.append(i_ind)
        dist.append(i_dist)
    ind = np.array(ind)
    
    if include_distances:
        return ind, np.array(dist)
    else:
        return ind

def MatchingBased_imp(modBatch_dict, output_key1='stage1_emb', target_key='stage2_emb', knn=2):
    test_mod_names = list(modBatch_dict.keys())
    n_test_batches = len(modBatch_dict[test_mod_names[0]])
    batch_contained_mod_ids = check_batch_empty(modBatch_dict, verbose=False)

    stage1_emb_pool, target_pool = {}, {}
    for k in test_mod_names:
        stage1_emb_pool[k], target_pool[k] = [], []
        for bi in range(n_test_batches):
            if modBatch_dict[k][bi] is not None:
                stage1_emb_pool[k].append(modBatch_dict[k][bi].obsm[output_key1])
                target_pool[k].append(
                    modBatch_dict[k][bi].X
                    if target_key=='X' else 
                    modBatch_dict[k][bi].obsm[target_key] 
                )

        stage1_emb_pool[k] = sps.vstack(stage1_emb_pool[k]).A if sps.issparse(stage1_emb_pool[k][0]) else np.vstack(stage1_emb_pool[k])
        target_pool[k]     = sps.vstack(target_pool[k]).A if sps.issparse(target_pool[k][0]) else np.vstack(target_pool[k])

    # perform imputation
    final_mod_data, full_data = [], {k: [None]*n_test_batches for k in test_mod_names}
    meta_perBatch, domain_labels, new_batch_labels = [], [], []
    for bi in range(n_test_batches):
        bi_valid_mod_names = [test_mod_names[v] for v in batch_contained_mod_ids[bi]] 
        bi_empty_mod_names = set(test_mod_names) - set(bi_valid_mod_names)
        for k in bi_empty_mod_names:
            imputed_dats = []
            for k2 in bi_valid_mod_names: # find matched profiles in another modal and taking average
                knn_ind = nn_approx(modBatch_dict[k2][bi].obsm[output_key1], stage1_emb_pool[k], knn=knn)
                matched_dats = target_pool[k][knn_ind.ravel()].reshape(knn_ind.shape[0], knn, target_pool[k].shape[1])
                imputed_dats.append(np.mean(matched_dats, axis=1))

            imputed_data = np.mean(imputed_dats, axis=0)
            full_data[k][bi] = imputed_data

        for k in bi_valid_mod_names:
            if target_key == 'X':
                full_data[k][bi] = modBatch_dict[k][bi].X
            else:
                full_data[k][bi] = modBatch_dict[k][bi].obsm[target_key] 

    return full_data, (n_test_batches, batch_contained_mod_ids, test_mod_names)

