import torch
import random
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sps
from torch.utils.data import Dataset
# from knn_classifier import nn_approx

random.seed(1)

class BaseDataset(Dataset):
    def __init__(
            self, feats, binz=True
        ):
        self.X = feats
        self.issparse = sps.issparse(feats)
        self.sample_num = self.X.shape[0]
        self.input_size = self.X.shape[1]
        self.binz = binz

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        sample = self.X[i].A if self.issparse else self.X[i]
        in_data = (sample>0).astype('float32') if self.binz else sample.astype('float32')

        return in_data.squeeze()


class NModalDataset(Dataset):
    def __init__(
            self, 
            dataset_dict
        ):
        self.mod_names = list(dataset_dict.keys())
        self.valid_mod_names = [k for k in self.mod_names if dataset_dict[k] is not None]
        self.dataset_dict = dataset_dict

    def __len__(self):
        return self.dataset_dict[self.valid_mod_names[0]].sample_num

    def __getitem__(self, i):
        samples = []  
        for mod in self.valid_mod_names:  
            samples.append(self.dataset_dict[mod].__getitem__(i))

        return i, samples
