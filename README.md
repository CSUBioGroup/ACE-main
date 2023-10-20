# ACE
ACE enhances representation of cellular heterogeneity and imputes missing molecular layers by mosaic integration.

## Installation
clone repository

`git clone https://github.com/CSUBioGroup/ACE-main.git`

`cd ACE-main/`

create env

`conda create -n ACE python=3.8.3`

`conda activate ACE`

install pytorch (our test-version: torch==1.12.1+cu116)

`pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116`

install other dependencies

`pip install -r requirements.txt`

Note that after installing `scib`, remember to compile using `g++ -std=c++11 -O3 knn_graph.cpp -o knn_graph.o` ([scib-issue](https://github.com/theislab/scib/issues/375)).

setup

`python setup.py install`

## Datasets
All datasets used in our paper can be found in [`zenodo`](). We provided the data used by our [`demo1`](./notebooks/demo_integration.ipynb) and [`demo2`](./notebooks/demo_imputation.ipynb) in the [`data/demo_data`](./data/demo_data) folder.

## Tutorials
In the [`notebooks`](./notebooks) folder, we provided notebooks to reproduce ACE's results in our manuscript.

[`sec1_sec2_integration`](./notebooks/sec1_sec2_integration) reproduced ACE's results in bimodal and trimodal integration benchmark.

[`sec3_imputation`](./notebooks/sec3_imputation) reproduced results in feature imputation. 

[`sec4_CITE-ASAP_refinement`](./notebooks/sec4_CITE-ASAP_refinement) reproduced experiments of annotation refinement analysis on CITE-ASAP dataset. 

[`sec5_COVID-19`](./notebooks/sec5_COVID-19) reproduced experiments on COVID-19 related datasets.

[`sec6_scalability`](./notebooks/sec6_scalability) tested ACE's time consumption. 

## Usage
Input format

For example, suppose that there are three batches to be integrated and they are measured within two modalities: RNA and adt (protein). Batch1 is measured with RNA and protein, batch2 measured with RNA only, and batch3 measured with protein only. These batches should be saved in a dictionary
```Python
modBatch_dict = {
    'rna': [batch1_rna, batch2_rna, None],
    'adt': [batch1_adt, None, batch3_adt] 
}
```
Keys 'rna' and 'adt' (or any other name) denote the modalities. Each key corresponds to a list of anndata objects which saves each batch's data from modality 'key' (rna or adt). Elements in the lists are in the same order of batches. `None` indicates the modality is not profiled in this batch. 

Another dictionary is used to specify where the input data is saved in the anndata object:
```Python
useReps_dict = {
    'rna': 'X_pca',
    'adt': 'X'
}
```
For each anndata in modality 'rna', their inputs are saved in the `.obsm['X_pca']`; for anndata in modality 'adt', their inputs are saved in `.X`. Note that these inputs will be directly given to the model for training, thus they should have been preprocessed.  

#### Stage 1
```Python
import math
from ACE.ace import ACE
# preparing inputs
modBatch_dict = {
    'rna': [batch1_rna, batch2_rna, None],    # following above case
    'adt': [batch1_adt, None, batch3_adt] 
}

useReps_dict = {
    'rna': 'X_pca',
    'adt': 'X'
}

T = 0.1    # temperature parameter
model = ACE(
    modBatch_dict=modBatch_dict,  
    useReps_dict=useReps_dict,  
    batch_key='batch', # column name of batch in the metadata, e.g., batch1_rna.obs['batch'] = 'batch1'
    layers_dims = {'rna': [1024, 512], 'adt':[512, 2048]}, # network params for each modality
    dropout_rates = {'rna':[0.2, 0.2], 'adt':[0.2, 0.2]},  # network params for each modality
    T=math.log(1./T), T_learnable=False, log_dir='outputs',
    n_latent1=256, n_latent2=256, seed=1234, num_workers=6 # most of hyper-params can be fixed 
)

model.stage1_fit(  # training 
    batch_size=512,
    epochs=100,
    lr=2e-4, 
    log_step=10, 
)

# stage1 inference
ad_integ = model.stage1_infer(
    modBatch_dict, useReps_dict, output_key='stage1_emb', 
    specify_mods_perBatch=[['rna'], ['rna'], ['adt']]
)
```

`specify_mods_perBatch` is a important parameter for stage1, which specifies which modalities are used for each batch to generate its embedding. In the example, we used batch1's rna modality to generate its embedding. Batch2 used rna and batch3 used adt. We can also specify `['adt']`, `['rna', 'adt']`, or `['all']` for batch 1. `['rna', 'adt']` and `['all']` mean using all the modalities contained in batch 1 to generate its embedding. In most scenarios of mosaic integration, we only specify one modality for each batch, because it can help balance batch correction and preservation of cellular heterogeneity. 

For feature imputation
```Python
# after stage1 training and inference
imputed_data = model.impute(modBatch_dict, output_key1='stage1_emb', knn=10, verbose=True)

# imputed_data is a dictionary like modBatch_dict, but its value is list of array
# e.g., {'rna':[rna_array1, rna_array2, rna_array3], 'adt':[adt_array1, adt_array2, adt_array3]}
batch2_adt_imputed = imputed_data['adt'][1]  # array 
batch3_rna_imputed = imputed_data['rna'][2]  # array
```

#### Stage 2
```Python
model.stage2_fit(
    batch_size=512,
    epochs=10,
    lr=1.75e-4, 
    log_step=5, 
    obvious_be=True,
)

# the inputs are the same as above
ad_integ2 = model.stage2_infer(
    modBatch_dict, useReps_dict, output_key1='stage1_emb', output_key2='stage2_emb', 
    knn=2, mod_weights={'rna':0.5, 'adt':0.5}
)
```

`knn` denotes the number of k-nearest neighbors used during performing cross-modal matching-based imputation. We set `knn=2` in most integration cases. `mod_weights` denotes the weight of each modality during averaging representations from all modalities. 