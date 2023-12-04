import torch
from torch import nn
import tinycudann as tcnn
import numpy as np

class NGP(nn.Module):
    def __init__(self, gray_act='Sigmoid'):
        super().__init__()

        self.gray_act = gray_act
        scale = 0.5
        L = 8; F = 2; log2_T = 10; N_min = 2
        b = np.exp(np.log(20*scale/N_min)/(L-1))
        
        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.gray_net = \
            tcnn.Network(
                n_input_dims=16, n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.gray_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

    def forward(self, x, **kwargs):
        h = self.xyz_encoder(x)
        grays = self.gray_net(h)


        return grays