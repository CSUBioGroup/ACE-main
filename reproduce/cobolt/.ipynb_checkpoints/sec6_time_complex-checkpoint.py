from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os
import gc
import math
import h5py
import datetime
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scanpy as sc
import scipy.io as sio

from os.path import join
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys


# data_dir = '/home/cb213/local/cache3/yxh/Data/seurat4-CITE-reference'
data_dir2 = '/home/yanxh/gitrepo/multi-omics-matching/tmp_outputs/time_complx/inputs'

print('Reading `mtx` files...')
_path = '/home/sda1/yanxh/data/seurat-CITE-reference/cite.h5'
with h5py.File(_path, 'r') as f:
    cell_names = np.array(f['cellID'], dtype='S32').astype('str')
#     rna_norm_data = sps.csc_matrix(
#             (np.array(f['RNA.data'], dtype=np.float32), 
#              np.array(f['RNA.indices'], dtype=np.int32),
#              np.array(f['RNA.indptr'], dtype=np.int32)
#             ), 
#             shape = np.array(f['RNA.shape'], dtype=np.int32)
#     ).tocsc().astype(np.float32).T#.toarray()
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

ad_cite.obsm['adt'] = adt_count_data


for rate in [0.01, 0.1, 0.2, 0.4, 0.8, 1.0]:
    smp_names = pd.read_csv(join(data_dir2, f'names_{rate}.csv'))['0'].values
    n_smp = len(smp_names)
    n_interval = n_smp // 3
    n_interval = n_interval-1 if n_interval % 128 == 1 else n_interval  # embarrassing
    ad_cite_subset = ad_cite[smp_names].copy()
    batch = ad_cite_subset.obs.batch.to_numpy()

    mult_rna_count = sps.csr_matrix(ad_cite_subset[:n_interval].X)
    mult_adt_count = sps.csr_matrix(ad_cite_subset[:n_interval].obsm['adt'])
    mult_barcode = ad_cite_subset.obs_names[:n_interval].to_numpy()
    rna_feature  = hvg_names
    adt_feature  = protein_names

    single_rna_count = sps.csr_matrix(ad_cite_subset[n_interval:int(2*n_interval)].X)
    single_adt_count = sps.csr_matrix(ad_cite_subset[int(2*n_interval):int(3*n_interval)].obsm['adt'])
    single_rna_barcode = ad_cite_subset.obs_names[n_interval:int(2*n_interval)].to_numpy()
    single_adt_barcode = ad_cite_subset.obs_names[int(2*n_interval):int(3*n_interval)].to_numpy()

    start_time = datetime.datetime.now()
    mult_rna = SingleData("GeneExpr", "Multiome", rna_feature, mult_rna_count, mult_barcode)
    mult_adt = SingleData("ADT", "Multiome", adt_feature, mult_adt_count, mult_barcode)

    single_rna = SingleData("GeneExpr", "Single-GEX", rna_feature, single_rna_count, single_rna_barcode)
    single_adt = SingleData("ADT", "Single-ADT", adt_feature, single_adt_count, single_adt_barcode)
    multi_dt = MultiomicDataset.from_singledata(
        single_rna, single_adt, mult_adt, mult_rna
    )
    print(f'mult_rna_count={mult_rna_count.shape}, mult_adt_count={mult_adt_count.shape}')
    print(f'single_rna_count={single_rna_count.shape}, single_adt_count={single_adt_count.shape}')
    
    model = Cobolt(dataset=multi_dt, lr=0.0001, n_latent=10, batch_size=120) # all default
    model.train(num_epochs=100)
    model.calc_all_latent()
    latent = model.get_all_latent()

    end_time = datetime.datetime.now()
    print('================================')
    print(f'Rate={rate}')
    print('Time cost: ', (end_time-start_time).total_seconds())
    print('================================')