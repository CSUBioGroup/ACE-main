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

import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.nn.functional as F

from src.networks import Encoder, AlignNet
from src.datasets import BaseDataset, NModalDataset
from src.loss import Npos_NCE
# from src.utils import save_model, load_model, stage1_fit_perStep, stage2_fit_perStep
import src.utils as utls

class ACE(object):
    def __init__(
        self, 
        modBatch_dict={},  # dict={'rna':[batch1, batch2, ...], 'adt':[batch1, batch2, ...], ...}
        useReps_dict={},   #  dict={'rna': 'X', 'adt': 'X', ...}
        batch_key='batch', 
        layers_dims = {'rna': [1024, 512], 'adt':[512, 2048], 'atac': [1024, 512]},
        dropout_rates = {'rna':[0.2, 0.2], 'adt':[0.2, 0.2],  'atac': [0.2, 0.2]},
        T=0.01, T_learnable=False, log_dir=None,
        n_latent1=40, n_latent2=256, seed=1234, num_workers=6
    ):  
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        self.num_workers = num_workers

        self.modBatch_dict = modBatch_dict
        self.useReps_dict = useReps_dict
        self.mod_names = np.array(list(modBatch_dict.keys()))
        self.n_mods = len(self.mod_names)
        self.n_batches = len(modBatch_dict[self.mod_names[0]])
        self.batch_key = batch_key

        self.mod_inputs, self.mod_inputSizes = utls._get_input(modBatch_dict, useReps_dict)
        self.batch_contained_mod_ids = utls.check_batch_empty(modBatch_dict)

        # check if this dataset can be integrated
        self.check_integrity()

        # network parameters
        self.layers_dims = layers_dims
        self.dropout_rates = dropout_rates
        self.n_latent1 = n_latent1
        self.n_latent2 = n_latent2
        self.init_T = T
        self.T_learnable = T_learnable

        self.build_stage1_net()
        self.build_stage2_net()

    def check_integrity(self):
        mod_graph = np.zeros((self.n_mods, self.n_mods))
        for bi in range(self.n_batches):
            modIds_in_bi = self.batch_contained_mod_ids[bi]
            mod_pairs = np.array(list(itertools.product(modIds_in_bi, modIds_in_bi)))
            mod_graph[mod_pairs[:, 0], mod_pairs[:, 1]] = 1
        n_cs, labels = connected_components(mod_graph, directed=False, return_labels=True)
        if n_cs > 1:
            for ci in np.unique(labels):
                ni_msk = labels == ci
                print(f'conn {ci}:', self.mod_names[ni_msk])
            raise RuntimeError('Dataset not connected, cannot be integrated')

    def build_stage1_net(self):
        self.stage1_encoders = AlignNet(
                                    Encoder, self.layers_dims, self.dropout_rates, 
                                    self.mod_inputSizes, 
                                    self.n_latent1, self.init_T
                                ).cuda()

        self.fixed_T1 = torch.tensor([self.init_T]).cuda() 

    def _build_stage1_dataset(self):
        datasets, train_batch_ids = [], []
        for bi in range(self.n_batches):
            n_valid_mods = len(self.batch_contained_mod_ids[bi])
            if n_valid_mods >= 2:  # build one dataset from this batch
                _ds = {}
                for mod in self.mod_names:  # make sure ds format consistent
                    X = self.mod_inputs[mod][bi]
                    _ds[mod] = None if X is None else BaseDataset(X, binz=False)
                datasets.append(NModalDataset(_ds))
                train_batch_ids.append(bi)
        print('Batch ids used in training: ', train_batch_ids)
        return datasets, train_batch_ids

    def build_stage2_net(self):
        self.stage2_encoders = {
            k: Encoder(
                n_input = self.mod_inputSizes[k],
                embedding_size=self.n_latent2,
                dropout_rates=self.dropout_rates[k],
                dims_layers=self.layers_dims[k]
            ).cuda()
            for k in self.mod_names
        }
        self.fixed_T2 = torch.tensor([self.init_T]).cuda()

    def _build_stage2_dataset(self, bridge_only=True):
        ds = {}
        use_batch_ids = self.bridge_batch_ids if bridge_only else range(self.n_batches)
        for mod in self.mod_names:
            ds[mod] = [
                BaseDataset(self.mod_inputs[mod][bi], binz=False) for bi in use_batch_ids
                if self.mod_inputs[mod][bi] is not None
            ]
        return ds

    def stage1_fit(
        self, 
        batch_size=512,
        epochs=100,
        lr=1e-4, 
        log_step=10, 
    ):  
        # dataset 
        datasets, bridge_batch_ids = self._build_stage1_dataset()
        self.bridge_batch_ids = bridge_batch_ids
        
        # in case empty training batch
        min_len = np.min([len(_) for _ in datasets])
        if min_len < batch_size:
            batch_size = 2**(int(math.log2(min_len)))
            print(f'decreasing batch size to {batch_size}')
            if batch_size < 32:
                raise ValueError(f'Certain batch is too small, minimum size should be no less than 32')
        
        # dl
        dls = []
        for _ds in datasets:
            dls.append(
                dataloader.DataLoader(_ds, batch_size=batch_size, 
                                      shuffle=True, num_workers=self.num_workers, drop_last=True)
            ) # have to set n_worker=0, otherwise leads to a unresolved bug of ipykernel

        # losses
        n_mods_inBatch = np.unique([len(_) for _ in self.batch_contained_mod_ids])
        n_mods_inBatch = n_mods_inBatch[n_mods_inBatch >= 2]
        crit_for_n_mods = {n:Npos_NCE(batch_size, rep=n).cuda() for n in n_mods_inBatch}

        opt = optim.Adam(self.stage1_encoders.parameters(), lr=lr, weight_decay=0.)
        T = self.stage1_encoders.T if self.T_learnable else self.fixed_T1

        loss_hist, metr_hist = defaultdict(list), defaultdict(list)
        step = 0
        print('Training stage1')
        for i in range(epochs):
            self.stage1_encoders.train()

            step, EPOCH_TOTAL_LOSS, EPOCH_BATCH_LOSSES = utls.stage1_fit_perStep(self.stage1_encoders, opt, crit_for_n_mods, datasets, dls, T)

            EPOCH_TOTAL_LOSS /= step
            loss_hist['total_loss'].append(EPOCH_TOTAL_LOSS)
            loss_hist['batch_losses'].append(EPOCH_BATCH_LOSSES)

            if (i+1) % log_step == 0:
                st = f'Epoch {i:d}, loss={EPOCH_TOTAL_LOSS:.4f}, temp={T.exp().item():.4f}, '
                for bi, tr_batch in enumerate(bridge_batch_ids):
                    st += f'batch{tr_batch} loss={EPOCH_BATCH_LOSSES[bi]:.4f}, '
                print(st)

                # saving model weight
                utls.save_model(
                    model=self.stage1_encoders.encoders, mod_names=self.mod_names, ckpt_id=i+1, 
                    log_dir=self.log_dir, stage_label='stage1'
                )

        self.stage1_loss_hist = loss_hist
        # self.stage1_metr_hist = metr_hist

    def stage2_fit(
        self, 
        batch_size=512,
        epochs=100,
        lr=1e-4, 
        log_step=10, 
        obvious_be=True
    ):
        datasets = self._build_stage2_dataset(bridge_only=not obvious_be)
        dataset_lens = [[len(_ds) for _ds in _dss] for mod, _dss in datasets.items()]
        min_len = np.min(np.ravel(dataset_lens))
        if min_len < batch_size:
            raise ValueError(f'Batch size too large, no greater than {min_len}')
        
        dls = {
           mod:[dataloader.DataLoader(_ds, batch_size=batch_size, 
                                     shuffle=True, num_workers=self.num_workers, drop_last=True) for _ds in _dss]
           for mod, _dss in datasets.items()
        }
        crit = nn.CrossEntropyLoss()
        opt = {k:optim.Adam(self.stage2_encoders[k].parameters(), lr=lr, weight_decay=0.) for k in self.mod_names}
        T = self.fixed_T2

        print('Training stage2')
        loss_hist = defaultdict(list)
        for i in range(epochs):
            # set mode
            for k in self.mod_names:
                self.stage2_encoders[k].train()

            MOD_EPOCH_LOSS = utls.stage2_fit_perStep(
                self.mod_names, self.stage2_encoders, opt, crit, datasets, dls, T
            )

            for k in self.mod_names:
                loss_hist[k].append(MOD_EPOCH_LOSS[k])

            if (i==0) or (i+1) % log_step == 0:
                st = f'Epoch {i:d}, '
                for k in self.mod_names:
                    st += f'{k}-loss={MOD_EPOCH_LOSS[k]:.4f}, '
                print(st)

                utls.save_model(
                    model=self.stage2_encoders, mod_names=self.mod_names, ckpt_id=i+1, 
                    log_dir=self.log_dir, stage_label='stage2'
                )
        self.stage2_loss_hist = loss_hist


    def stage1_encoding(self, modBatch_dict, useReps_dict, output_key='stage1_emb', batch_size=512):
        self.stage1_encoders.eval()
        test_mod_names = list(modBatch_dict.keys())
        n_test_batches = len(modBatch_dict[test_mod_names[0]])
        batch_contained_mod_ids = utls.check_batch_empty(modBatch_dict, verbose=False)
        assert set(test_mod_names) & set(self.mod_names) == set(test_mod_names), 'Unrecognized modalities'

        test_mod_inputs, _ = utls._get_input(modBatch_dict, useReps_dict)
        for bi in range(n_test_batches):
            _ds = {}
            for mod in test_mod_names:
                if modBatch_dict[mod][bi] is None:
                    _ds[mod] = None 
                else:
                    _ds[mod] = BaseDataset(test_mod_inputs[mod][bi], binz=False)
            ds = NModalDataset(_ds)
            dl = dataloader.DataLoader(ds, batch_size=batch_size, 
                                       shuffle=False, num_workers=self.num_workers, drop_last=False)

            mod_feats = {mod:[] for mod in ds.valid_mod_names}  # for current batch
            for _, batch_data in dl:
                batch_data = [_.cuda() for _ in batch_data]
                batch_dict = dict(zip(ds.valid_mod_names, batch_data))
                
                feat_dict = self.stage1_encoders(batch_dict)
                for mod in ds.valid_mod_names:
                    mod_feats[mod].append(feat_dict[mod].detach().cpu().numpy())

            for mod, feat_list in mod_feats.items():
                modBatch_dict[mod][bi].obsm[output_key] = np.vstack(feat_list)

        return n_test_batches, test_mod_names, batch_contained_mod_ids, modBatch_dict

    def stage2_encoding(self, modBatch_dict, useReps_dict, output_key='stage2_emb', batch_size=512):
        # set mode
        for k in self.mod_names:
            self.stage2_encoders[k].eval()

        test_mod_names = list(modBatch_dict.keys())
        n_test_batches = len(modBatch_dict[test_mod_names[0]])
        batch_contained_mod_ids = utls.check_batch_empty(modBatch_dict, verbose=False)
        assert set(test_mod_names) & set(self.mod_names) == set(test_mod_names), 'Unrecognized modalities'

        test_mod_inputs, _ = utls._get_input(modBatch_dict, useReps_dict)
        for k in self.mod_names:
            for bi in range(n_test_batches):
                if test_mod_inputs[k][bi] is not None:
                    _ds = BaseDataset(test_mod_inputs[k][bi], binz=False)
                    _dl = dataloader.DataLoader(_ds, batch_size=batch_size, 
                                   shuffle=False, num_workers=self.num_workers, drop_last=False)

                    embedding = []
                    for _data in _dl:
                        _data = _data.float().cuda()
                        _feat = F.normalize(self.stage2_encoders[k](_data), dim=1, p=2)
                        embedding.append(_feat.detach().cpu().numpy())
                    modBatch_dict[k][bi].obsm[output_key] = np.vstack(embedding)

        return n_test_batches, test_mod_names, batch_contained_mod_ids, modBatch_dict

    def stage1_infer(self, modBatch_dict, useReps_dict, output_key='stage1_emb', specify_mods_perBatch=[['rna'], ['rna'], ['adt']]):
        n_test_batches, test_mod_names, batch_contained_mod_ids, modBatch_dict = self.stage1_encoding(
            modBatch_dict, useReps_dict, output_key=output_key
        )
        assert n_test_batches == len(specify_mods_perBatch), 'unmatched shape'
        assert (set(np.ravel(specify_mods_perBatch)) - set(['all'])).issubset(set(test_mod_names)), 'Unrecognized modalities'

        final_emb_perBatch, meta_perBatch = [], []
        for bi, used_mods in enumerate(specify_mods_perBatch):
            bi_embs = []
            visit_mods = test_mod_names if 'all' in used_mods else used_mods
            for mod in visit_mods:
                if modBatch_dict[mod][bi] is not None:
                    bi_embs.append(modBatch_dict[mod][bi].obsm[output_key])
            final_emb_perBatch.append(np.mean(bi_embs, axis=0))

        domain_labels, new_batch_labels = [], []
        for bi in range(n_test_batches):
            for mod in test_mod_names:
                if modBatch_dict[mod][bi] is not None:
                    meta_perBatch.append(modBatch_dict[mod][bi].obs.copy())
                    n_samples = modBatch_dict[mod][bi].shape[0]
                    break
            domain_labels.extend(
                [
                    '+'.join([test_mod_names[_] for _ in batch_contained_mod_ids[bi]])
                ]*n_samples
            )
            # new_batch_labels.extend([f'batch{bi+1}']*n_samples)

        ad_integ = sc.AnnData(np.vstack(final_emb_perBatch), obs=pd.concat(meta_perBatch))
        ad_integ.obs['raw_batch']    = ad_integ.obs[self.batch_key].to_numpy()
        ad_integ.obs['domain']       = domain_labels
        ad_integ.obs[self.batch_key] = (ad_integ.obs.domain + '-' + ad_integ.obs.raw_batch).to_numpy()
        return ad_integ

    def stage2_infer(self, modBatch_dict, useReps_dict, output_key1='stage1_emb', output_key2='stage2_emb', 
            knn=2, mod_weights={'rna':0.5, 'adt':0.5}
        ):
        n_test_batches, test_mod_names, batch_contained_mod_ids, modBatch_dict = self.stage2_encoding(
            modBatch_dict, useReps_dict, output_key=output_key2
        )

        # cross-modal matching-based imputation 
        full_stage2_embs, _ = utls.MatchingBased_imp(modBatch_dict, output_key1=output_key1, target_key=output_key2, knn=knn)

        # take a average
        final_mod_embs, meta_perBatch, domain_labels, new_batch_labels = [], [], [], []
        for bi in range(n_test_batches):
            contained_mods = [test_mod_names[_] for _ in batch_contained_mod_ids[bi]]
            # averaging over all modalities
            bi_final_emb = full_stage2_embs[test_mod_names[0]][bi] * mod_weights[test_mod_names[0]]
            for k in test_mod_names[1:]:
                bi_final_emb += full_stage2_embs[k][bi] * mod_weights[k]

            final_mod_embs.append(bi_final_emb)
            meta_perBatch.append(modBatch_dict[contained_mods[0]][bi].obs.copy())
            domain_labels.extend(['+'.join(contained_mods)]*bi_final_emb.shape[0])
            # new_batch_labels.extend([f'batch{bi+1}']*bi_final_emb.shape[0])

        ad_integ = sc.AnnData(np.vstack(final_mod_embs), obs=pd.concat(meta_perBatch))
        ad_integ.obs['raw_batch'] = ad_integ.obs[self.batch_key].to_numpy()
        ad_integ.obs['domain']    = domain_labels
        ad_integ.obs[self.batch_key] = (ad_integ.obs.domain + '-' + ad_integ.obs.raw_batch).to_numpy()
        return ad_integ

    def impute(self, modBatch_dict, output_key1='stage1_emb', knn=2, verbose=True):
        full_X, (n_test_batches, batch_contained_mod_ids, test_mod_names) = utls.MatchingBased_imp(
            modBatch_dict, output_key1=output_key1, target_key='X', knn=knn
        )

        if verbose:
            for bi in range(n_test_batches):
                valid_mod_names = [test_mod_names[_] for _ in batch_contained_mod_ids[bi]]
                empty_mod_names = list(set(test_mod_names) - set(valid_mod_names))
                if len(empty_mod_names) >= 1:
                    print(f'batch {bi}: impute => {empty_mod_names}')
                else:
                    print(f'batch {bi}: no need for imputation')
                    
        return full_X















    