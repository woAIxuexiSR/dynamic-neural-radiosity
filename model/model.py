import drjit as dr
import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
import time

grid_config = {
    "otype": "HashGrid",
    "n_dims_to_encode": 3,
    "n_levels": 8,
    "n_features_per_level": 8,
    "log2_hashmap_size": 19,
    "base_resolution": 32,
    "per_level_scale": 2.0,
    "interpolation": "Linear",
}

direction_normal_config = {
    "otype": "SphericalHarmonics",
    "n_dims_to_encode": 3,
    "degree": 4,
}

reflectance_config = {
    "otype": "Identity",
    "n_dims_to_encode": 3,
}

roughness_config = {
    "otype": "OneBlob",
    "n_dims_to_encode": 1,
    "n_bins": 8 
}

network_config = {
    "otype": "CutlassMLP",
    # "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

vvmlp_encoding_config = {
	"otype": "Frequency",
    "n_dims_to_encode": 1,
	"n_frequencies": 8,
}

vvmlp_network_config = {
    # "otype": "CutlassMLP",
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

class DNR(nn.Module):

    def __init__(self, bbox, config: dict, var_num):
        super().__init__()
        self.bb_min = bbox.min
        self.bb_max = bbox.max
        self.var_num = var_num
        self.encoding_reduce = config.get("encoding_reduce", "concatenation")
        assert self.encoding_reduce in ["concatenation", "sum", "product"]

        encoding_config = {"otype": "Composite", "nested": []}
        encoding_config["nested"].append(grid_config)

        var_config = {
            "otype": config["grid_type"],
            "n_dims_to_encode": 2,
            "n_levels": config["n_levels"],
            "n_features_per_level": config["n_features_per_level"],
            "base_resolution": config["base_resolution"],
            "per_level_scale": config["per_level_scale"],
            "interpolation": "Linear",
        }

        # make an integrated encoding for all variables to define its reduce type
        var_encoding_config = {
            "otype": "Composite", 
            "reduction": self.encoding_reduce,
            "nested": []
        }
        for _ in range(var_num * 3):
            var_encoding_config["nested"].append(var_config)

        encoding_config["nested"].append(var_encoding_config)

        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(direction_normal_config)
        encoding_config["nested"].append(reflectance_config)

        network_config["n_neurons"] = config["n_neurons"]
        network_config["n_hidden_layers"] = config["n_hidden_layers"]

        self.n_input_dims = 3 + 3 * var_num * 2 + 3 + 3 + 3
        self.encoding = tcnn.Encoding(self.n_input_dims, encoding_config)

        self.vvmlp_output_dims = config.get("vvmlp_output_dims", 128)
        vvmlp_encoding_config["n_dims_to_encode"] = var_num
        self.vvmlp = tcnn.NetworkWithInputEncoding(var_num, self.vvmlp_output_dims, vvmlp_encoding_config, vvmlp_network_config)
        
        self.mlp = tcnn.Network(self.encoding.n_output_dims + self.vvmlp_output_dims, 3, network_config)
        
    def forward(self, pos, dirs, normal, albedo, vars: np.ndarray):


        with dr.suspend_grad():
            pos = ((pos - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = dirs.torch()
            n = normal.torch()
            f_d = albedo.torch()
            vars = torch.from_numpy(vars)
            
            # there are some nan values due to scene.ray_intersect()
            wi[wi.isnan()] = 0.0
            n[n.isnan()] = 0.0

            wi = (nn.functional.normalize(wi, dim=1) + 1) * 0.5
            n = (nn.functional.normalize(n, dim=1) + 1) * 0.5

        var_num = self.var_num
        x = torch.zeros((pos.shape[0], 3 + 3 * var_num * 2), device="cuda")
        x[:, :3] = pos
        for i in range(var_num):
            x[:, 3 + i * 6] = pos[:, 0]
            x[:, 3 + i * 6 + 1] = vars[i]
            x[:, 3 + i * 6 + 2] = pos[:, 1]
            x[:, 3 + i * 6 + 3] = vars[i]
            x[:, 3 + i * 6 + 4] = pos[:, 2]
            x[:, 3 + i * 6 + 5] = vars[i]
        x = torch.cat([x, wi, n, f_d], dim=1).to(torch.float32)
        
        x = self.encoding(x)

        v = torch.zeros((pos.shape[0], var_num), device="cuda")
        for i in range(var_num):
            v[:, i] = vars[i]
        v = self.vvmlp(v)

        x = self.mlp(torch.cat([x, v], dim=1)).to(torch.float32).abs()
        
        return x