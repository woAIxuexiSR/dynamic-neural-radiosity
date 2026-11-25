# Dynamic Neural Radiosity

The code release for the paper "Dynamic Neural Radiosity with Multi-grid Decomposition" (SIGGRAPH Asia 2024).

## Requirements

**Tested environment**

- Ubuntu 22.04
- Python 3.12.16
- CUDA Toolkit 12.6
- PyTorch 2.7.1 with CUDA 12.6
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 2.0
- Mitsuba 3.10.0.dev1, pinned to commit
  `aa01500f57770a4ef0b067693cae08b3e8a2ac73`
- Dr.Jit 1.6.0.dev1

The interactive viewer in `test.py` additionally requires `imgui`,
`PyOpenGL`, `pycuda`, and `glfw`. These packages are not required for
training or offline rendering with `render_img.py`.

## Cloning the Repository

Clone the repository together with the pinned Mitsuba source tree:

```bash
git clone --recurse-submodules <repository-url>
cd dynamic-neural-radiosity
```

For an existing clone:

```bash
git submodule update --init --recursive
```

The pretrained model is optional and is stored with Git LFS. It is only needed
when passing `-m pretrained_model/dining-room/model.pth`. To download it:

```bash
git lfs install
git lfs pull --include="pretrained_model/dining-room/model.pth"
```

To skip the pretrained model when Git LFS is already installed:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules <repository-url>
```

## Compiling Mitsuba 3 from Source

This project uses the `cuda_rgb` variant of Mitsuba 3, which is not available in pre-built binaries. You need to compile it from the [source](mitsuba) included as a submodule.

Refer to the [official documentation](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html) for building on different platforms. On Windows:

```bash
cd mitsuba
cmake -G "Visual Studio 17 2022" -A x64 -B build
```

Enable the `cuda_rgb` variant in `build/mitsuba.conf` (~line 86), then build:

```bash
cmake --build build --config Release
```

After building, run the `setpath` script to configure environment variables (`PATH`, `PYTHONPATH`):

```bash
# Windows PowerShell
cd mitsuba\build\Release
.\setpath.ps1

# Linux
source mitsuba/build/setpath.sh
```

## Usage

**Train a model:**

```bash
python train.py [-c config.json] [-m pretrained_model.pth] [--profile-steps steps] [--jit-fusion-encoding | --no-jit-fusion-encoding]
```

- `--profile-steps`: run only the specified number of steps and print profiling statistics
- `--jit-fusion-encoding` / `--no-jit-fusion-encoding`: enable or disable tiny-cuda-nn JIT fusion for the encoding; enabled by default

**Interactive viewer:**

```bash
python test.py [-c config.json] [-m pretrained_model.pth] [--jit-fusion-encoding | --no-jit-fusion-encoding]
```

**Render an image:**

```bash
python render_img.py [-t type] [-s spp] [-c config.json] [-m pretrained_model.pth] [-o output.exr] [--profile] [--jit-fusion-encoding | --no-jit-fusion-encoding]
```

- `-t`: rendering method — `LHS`, `RHS`, or `path`
- `-s`: samples per pixel
- `-o`: output file path
- `--profile`: print synchronized render profiling statistics
- `--jit-fusion-encoding` / `--no-jit-fusion-encoding`: enable or disable tiny-cuda-nn JIT fusion for the encoding; enabled by default

> **Note:** The FXAA post-processing used in the paper is not included. See [glsl-fxaa](https://github.com/mattdesl/glsl-fxaa) for a reference implementation.

## Config File

| Field | Description |
|---|---|
| `scene` | Path to the Mitsuba scene file (`.xml`) |
| `animation` | Path to the animation file (`.json`), see [scenes/README.md](scenes/README.md) |
| `v` | Animation variable values, or `""` for random |
| `output` | Output directory for saved models |

**Training parameters** (`train`):

| Field | Description |
|---|---|
| `use_adaptive_rhs` | Progressively increase RHS samples during training |
| `loss` | Loss function (e.g., `normed_semi_l2`) |
| `rhs_samples` | Number of samples for the right-hand side |
| `batch_size` | Batch size |
| `steps` | Number of training steps |
| `learning_rate` | Learning rate |
| `loss_interval` | TensorBoard loss logging interval (in steps) |
| `save_interval` | Model checkpoint interval (in steps) |
| `model_name` | Checkpoint filename |

**Model parameters** (`model`):

| Field | Description |
|---|---|
| `type` | Model type (`DNR`) |
| `grid_type` | Grid type (`DenseGrid` or `HashGrid`) |
| `n_levels` | Number of grid levels |
| `n_features_per_level` | Features per grid level |
| `base_resolution` | Base resolution of the grid |
| `per_level_scale` | Resolution scale factor per level |
| `n_neurons` | Neurons per hidden layer |
| `n_hidden_layers` | Number of hidden layers |
| `vvmlp_output_dims` | Output dimensions of the variable MLP |
| `encoding_reduce` | Encoding reduction method (`concatenation`, `sum`, or `product`) |

<details>
<summary>Example config</summary>

```json
{
    "scene": "scenes/dining-room/scene.xml",
    "animation": "scenes/dining-room/animation.json",
    "v": [0.5, 0.4, 0.6, 0.7, 0.2, 0.8, 0.5],
    "output": "dining-room",
    "train": {
        "use_adaptive_rhs": true,
        "loss": "normed_semi_l2",
        "rhs_samples": 32,
        "batch_size": 16384,
        "steps": 120000,
        "learning_rate": 0.001,
        "loss_interval": 500,
        "save_interval": 10000,
        "model_name": "model.pth"
    },
    "model": {
        "type": "DNR",
        "grid_type": "HashGrid",
        "n_levels": 4,
        "n_features_per_level": 2,
        "base_resolution": 32,
        "per_level_scale": 2.0,
        "n_neurons": 256,
        "n_hidden_layers": 4,
        "vvmlp_output_dims": 128,
        "encoding_reduce": "concatenation"
    }
}
```

</details>
