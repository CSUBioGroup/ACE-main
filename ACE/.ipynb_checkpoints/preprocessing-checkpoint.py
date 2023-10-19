from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.utils.extmath
from harmony import harmonize

class tfidfTransformer:
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / X.sum(axis=0)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            return tf.multiply(self.idf)
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# optional, other reasonable preprocessing steps are also acceptable.
class lsiTransformer:
    def __init__(
        self, n_components: int = 20, drop_first=True, use_highly_variable=None, log=True, norm=True, z_score=True,
        tfidf=True, svd=True, use_counts=False, pcaAlgo='arpack'
    ):  

        self.drop_first = drop_first
        self.n_components = n_components + drop_first
        self.use_highly_variable = use_highly_variable

        self.log = log
        self.norm = norm
        self.z_score = z_score
        self.svd = svd
        self.tfidf = tfidf
        self.use_counts = use_counts

        self.tfidfTransformer = tfidfTransformer()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(
            n_components=self.n_components, random_state=777, algorithm=pcaAlgo
        )
        self.fitted = None

    def fit(self, adata: anndata.AnnData):
        if self.use_highly_variable is None:
            self.use_highly_variable = "highly_variable" in adata.var
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X = adata_use.layers['counts']
        else:
            X = adata_use.X
        if self.tfidf:
            X = self.tfidfTransformer.fit_transform(X)
        if scipy.sparse.issparse(X):
            X = X.A.astype("float32")
        if self.norm:
            X = self.normalizer.fit_transform(X)
        if self.log:
            X = np.log1p(X * 1e4)    # L1-norm and target_sum=1e4 and log1p
        self.pcaTransformer.fit(X)
        self.fitted = True

    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X_pp = adata_use.layers['counts']
        else:
            X_pp = adata_use.X
        if self.tfidf:
            X_pp = self.tfidfTransformer.transform(X_pp)
        if scipy.sparse.issparse(X_pp):
            X_pp = X_pp.A.astype("float32")
        if self.norm:
            X_pp = self.normalizer.transform(X_pp)
        if self.log:
            X_pp = np.log1p(X_pp * 1e4)
        if self.svd:
            X_pp = self.pcaTransformer.transform(X_pp)
        if self.z_score:
            X_pp -= X_pp.mean(axis=1, keepdims=True)
            X_pp /= X_pp.std(axis=1, ddof=1, keepdims=True)
        pp_df = pd.DataFrame(X_pp, index=adata_use.obs_names).iloc[
            :, int(self.drop_first) :
        ]
        return pp_df

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)
   
# optional     
class ADTransformer:
    def __init__(
        self, n_components: int = 20, drop_first=True, use_highly_variable=None, svd=True, norm=True, 
        log=True, z_score=True, use_counts=False
    ):

        self.drop_first = drop_first
        self.n_components = n_components + drop_first
        self.use_highly_variable = use_highly_variable
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(
            n_components=self.n_components, random_state=777, algorithm='arpack'
        )
        self.fitted = None
        self.svd = svd
        self.log = log
        self.norm = norm
        self.z_score = z_score
        self.use_counts = use_counts

    def fit(self, adata: anndata.AnnData):
        if self.use_highly_variable is None:
            self.use_highly_variable = "highly_variable" in adata.var
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X = adata.layers['counts'].A
        else:
            X = adata.X.A
        if self.norm:
            X = self.normalizer.fit_transform(X)
        if self.log:
            X = np.log1p(X * 1e4)    # L1-norm and target_sum=1e4 and log1p

        if self.svd:
            self.pcaTransformer.fit(X)
        self.fitted = True

    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X_pp = adata_use.layers['counts'].A
        else:
            X_pp = adata_use.X.A
        if self.norm:
            X_pp = self.normalizer.transform(X_pp)
        if self.log:
            X_pp = np.log1p(X_pp * 1e4)
        if self.svd:
            X_pp = self.pcaTransformer.transform(X_pp)
        if self.z_score:
            X_pp -= X_pp.mean(axis=1, keepdims=True)
            X_pp /= X_pp.std(axis=1, ddof=1, keepdims=True)
        pp_df = pd.DataFrame(X_pp, index=adata_use.obs_names).iloc[
            :, int(self.drop_first) :
        ]
        return pp_df

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)

def harmony(df_list, use_gpu=True):
    """
    Harmony batch effect correction applied jointly to multiple dataframes
    """
    all = pd.concat(df_list)
    all_batches = all.pop("batch")
    all_batches.columns = ["batch"]
    all_batches = all_batches.to_frame()
    all_harmony = harmonize(
        all.to_numpy(), all_batches, batch_key="batch", use_gpu=use_gpu, verbose=True
    )
    out_df_list = []
    curr_idx = 0
    for df in df_list:
        out_df_list.append(all_harmony[curr_idx : curr_idx + len(df)])
        curr_idx += len(df)
    return out_df_list

def HARMONY(all_df, batch_labels, use_gpu=True):
    all_df['batch'] = batch_labels
    all_arr = harmony([all_df])[0]
    return all_arr


