import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.weight_norm as weight_norm

dropout_rate = .2

class Encoder(nn.Module):
    def __init__(
        self,
        n_input,
        embedding_size,
        dropout_rates,
        dims_layers,
    ):
        super(Encoder, self).__init__()
        dropout = []
        layers = [nn.Linear(n_input, dims_layers[0])]

        for i in range(len(dims_layers) - 1):
            layers.append(nn.Linear(dims_layers[i], dims_layers[i + 1]))
        for i in range(len(dropout_rates)):
            dropout.append(nn.Dropout(p=dropout_rates[i]))

        layers.append(nn.Linear(dims_layers[-1], embedding_size))

        self.fc_list = nn.ModuleList(layers)
        # print("dropout list", dropout)
        self.dropout_list = nn.ModuleList(dropout)

    def forward(self, x):
        for i in range(len(self.fc_list) - 1):
            x = F.elu(self.fc_list[i](x))
            if i < len(self.dropout_list):
                x = self.dropout_list[i](x)

        x = self.fc_list[-1](x)
        return x


class AlignNet(nn.Module):
    """CLIP-inspired architecture"""

    def __init__(
        self,
        Encoder,
        layers_dims,  # dict 
        dropout_rates, # dict 
        input_dims,     # dict
        output_dim,
        T,
    ):
        super(AlignNet, self).__init__()

        self.n_mod = len(layers_dims)
        self.encoders = nn.ModuleDict()
        for mod, _ in layers_dims.items():
            self.encoders[mod] = Encoder(
                    input_dims[mod],
                    output_dim,
                    dropout_rates[mod],
                    layers_dims[mod]
                )

        self.T = nn.Parameter(torch.ones([]) * T)

    def forward(self, mod_feats={'rna': None, 'adt':None, 'atac':None}):
        output_feats = {}
        for k, v in mod_feats.items():
            if v is None:
                output_feats[k] = None
            else:
                output_feats[k] = F.normalize(self.encoders[k](mod_feats[k]), p=2, dim=1)

        return output_feats
