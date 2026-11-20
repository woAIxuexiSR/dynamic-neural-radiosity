import drjit as dr
import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
from contextlib import nullcontext

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

        self.jit_fusion_requested = bool(
            config.get("jit_fusion_encoding", True)
        )
        self.jit_fusion_supported = False
        self.jit_fusion_modules = []
        if self.jit_fusion_requested:
            self.jit_fusion_supported = bool(tcnn.supports_jit_fusion())
            if self.jit_fusion_supported:
                self.encoding.jit_fusion = True
                self.jit_fusion_modules.append("encoding")

        print(
            "tiny-cuda-nn encoding JIT fusion: "
            f"requested={self.jit_fusion_requested}, "
            f"supported={self.jit_fusion_supported}, "
            f"enabled={self.jit_fusion_modules or 'none'}",
            flush=True,
        )
        
    def _forward_torch(
        self,
        pos,
        wi,
        n,
        f_d,
        vars: np.ndarray,
        profiler=None,
    ):
        timer = (
            profiler.measure
            if profiler is not None
            else lambda name: nullcontext()
        )

        vars = torch.as_tensor(vars, device="cuda", dtype=torch.float32)

        # There are some NaN values due to scene.ray_intersect().
        wi = wi.nan_to_num(0.0)
        n = n.nan_to_num(0.0)

        wi = (nn.functional.normalize(wi, dim=1) + 1) * 0.5
        n = (nn.functional.normalize(n, dim=1) + 1) * 0.5

        var_num = self.var_num
        pos_per_var = pos[:, None, :].expand(-1, var_num, -1)
        var_per_pos = vars.view(1, var_num, 1).expand(pos.shape[0], -1, 3)
        variable_input = torch.stack(
            (pos_per_var, var_per_pos), dim=-1
        ).reshape(pos.shape[0], var_num * 6)
        x = torch.cat([pos, variable_input, wi, n, f_d], dim=1)

        v = vars.view(1, var_num).expand(pos.shape[0], -1).contiguous()

        with timer("model.encoding"):
            x = self.encoding(x)

        v = self.vvmlp(v)

        with timer("model.head_mlp"):
            x = self.mlp(torch.cat([x, v], dim=1)).to(torch.float32).abs()

        return x

    def forward(
        self,
        pos,
        dirs,
        normal,
        albedo,
        vars: np.ndarray,
        profiler=None,
    ):
        timer = (
            profiler.measure
            if profiler is not None
            else lambda name: nullcontext()
        )

        with timer("model.drjit_to_torch"):
            with dr.suspend_grad():
                pos = (pos - self.bb_min) / (self.bb_max - self.bb_min)
                packed = dr.stack(
                    [
                        pos.x,
                        pos.y,
                        pos.z,
                        dirs.x,
                        dirs.y,
                        dirs.z,
                        normal.x,
                        normal.y,
                        normal.z,
                        albedo.x,
                        albedo.y,
                        albedo.z,
                    ],
                    axis=0,
                ).torch()
                pos, wi, n, f_d = (
                    packed.reshape(4, 3, -1)
                    .transpose(1, 2)
                    .unbind(0)
                )

        return self._forward_torch(
            pos,
            wi,
            n,
            f_d,
            vars,
            profiler,
        )
