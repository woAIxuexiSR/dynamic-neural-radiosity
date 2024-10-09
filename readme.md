# Dynamic Neural Radiosity

The code release for the paper "Dynamic Neural Radiosity with Multi-grid Decomposition" (SIGGRAPH Asia 2024).

## Platform and Dependencies

* Windows 10 / Ubuntu 20.04

* Python 3.10
* PyTorch 2.1.0
* tinycudann (https://github.com/NVlabs/tiny-cuda-nn.git)
* numpy
* imgui
* pyopengl
* pycuda

## Compiling mitsuba3 from source

This is necessary because the variant of mitsuba3 used in this project is "cuda_rgb" which is not available in the pre-built binaries.

To use custom functions for mitsuba3, you need to compile it from [source](mitsuba), and set corresponding environment variables (`PATH/PYTHONPATH`) that are required to run Mitsuba.

Refer to [the document](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html) for building on different platforms. Specifically, on Windows, one can build Mitsuba using:

``` bash
    cd mitsuba
    cmake -G "Visual Studio 17 2022" -A x64 -B build
```

Change the `mitsuba.conf` file in the `build` directory to enable the `cuda_rgb` variant(about lin 86). Then build Mitsuba using:

``` bash
    cmake --build build --config Release
```

Once Mitsuba is built, run the `setpath.sh/.bat/.ps1` script in your build directory to configure environment variables (`PATH/PYTHONPATH`) that are required to run Mitsuba. E.g., on Windows Powershell:

``` bash
    cd mitsuba\build\Release    
    .\setpath.ps1
```

## Animations

1. For obj shapes, the transformation only support one type.
2. For others, the transformation can be composed by translation and scale. Rotation can't be composed.

For example:

``` json

"translate": [
    {
        "shape_names": ["Ball"],
        "start": [0.2, 1.0, 0],
        "end": [-0.4, 0.4, 0]
    }
],
"scale": [
    {
        "shape_names": ["Ball"],
        "start": 0.4,
        "end": 0.2
    }
],
```

If an animation contains many shapes, all the shapes should be written in the `shape_names` list. The `shape_names` list must contain only obj shapes or only others.

## Config File

* `scene`: the path of the scene file.
* `animation`: the path of the animation file.
* `output`: the output directory of the model.
* `train`: the training parameters.
  * `use_adaptive_rhs`: whether to use adaptive rhs.
  * `loss`: loss function.
  * `rhs_samples`: the number of samples for the right hand side.
  * `batch_size`: the batch size.
  * `steps`: the number of training steps.
  * `learning_rate`: the learning rate.
  * `save_interval`: the interval of saving the model.
  * `model_name`: the name of the model.
* `model`: the model parameters.
  * `type`: the type of the model, support `DNR`.
  * `grid_type`: the type of the grid, support `DenseGrid`, `HashGrid`.
  * `n_levels`: the number of levels of the grid.
  * `n_features_per_level`: the number of features per level.
  * `base_resolution`: the base resolution of the grid.
  * `per_level_scale`: the scale of the resolution of each level.
  * `n_neurons`: the number of neurons in each hidden layer.
  * `n_hidden_layers`: the number of hidden layers.
  * `vvmlp_output_dims`: the output dimensions of the vvmlp.
  * `encoding_reduce`: the reduce function of the encoding layer.

For example:

``` json
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
        "save_interval": 500,
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

## Run

Note: The FXAA used in paper is not implemented in this project, where you can find here: https://github.com/mattdesl/glsl-fxaa.git.

* train a model: `python train.py [-c config.json] [-m pre_trained_model]`
* test ui: `python test.py [-c config.json] [-m pre_trained_model]`
* render an image:  
  `python render_img.py [-t type] [-s spp] [-c config.json] [-m pre_trained_model] [-o output.exr]`
  * `type`: the type of the rendering method, support `LHS`, `RHS`, `path`
  * `spp`: the number of samples per pixel
  * `output`: the output file path