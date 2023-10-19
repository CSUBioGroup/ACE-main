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
