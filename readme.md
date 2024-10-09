# Dynamic Neural Radiosity

## Platform and Dependencies

* Windows 10 / Ubuntu 20.04

* Python 3.10
* PyTorch 2.1.0
* tinycudann (https://github.com/NVlabs/tiny-cuda-nn.git)
* numpy
* imgui
* pyopengl
* pycuda

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
  * `use_mcmc`: whether to use MCMC to sample the variable space.
  * `use_adaptive_surface`: whether to use adaptive surface to sample.
  * `factorize_diffspec`: whether to factorize the diffuse and specular components for network output.
  * `rhs_samples`: the number of samples for the right hand side.
  * `batch_size`: the batch size.
  * `steps`: the number of training steps.
  * `learning_rate`: the learning rate.
  * `save_interval`: the interval of saving the model.
  * `model_name`: the name of the model.
* `model`: the model parameters.
  * `type`: the type of the model, support `DNRField`, `DNRFieldFull`.
  * `grid_type`: the type of the grid, support `DenseGrid`, `HashGrid`.
  * `n_levels`: the number of levels of the grid.
  * `n_features_per_level`: the number of features per level.
  * `base_resolution`: the base resolution of the grid.
  * `per_level_scale`: the scale of the resolution of each level.
  * `n_neurons`: the number of neurons in each hidden layer.
  * `n_hidden_layers`: the number of hidden layers.
  * `factorize_reflectance`: whether to factorize the reflectance from network output.

For example:

``` json
{
    "scene": "scenes/sphere-caustic/scene.xml",
    "animation": "scenes/sphere-caustic/animation.json",
    "output": "sphere-caustic",
    "train": {
        "use_mcmc": true,
        "use_adaptive_surface": true,
        "rhs_samples": 32,
        "batch_size": 16384,
        "steps": 100000,
        "learning_rate": 0.001,
        "save_interval": 500,
        "model_name": "model.pth"
    },
    "model": {
        "type": "DNRField",
        
        "grid_type": "DenseGrid",
        "n_levels": 4,
        "n_features_per_level": 8,
        "base_resolution": 32,
        "per_level_scale": 2.0,

        "n_neurons": 256,
        "n_hidden_layers": 4
    }
}
```

## Run

* train a model: `python main.py [-c config.json] [-m pre_trained_model]`
* test: `python test.py [-c config.json] [-m pre_trained_model]`
* render an image:  
  `python render_img.py [-t type] [-s spp] [-c config.json] [-m pre_trained_model] [-o output.exr]`
  * `type`: the type of the rendering method, support `LHS`, `RHS`, `path`
  * `spp`: the number of samples per pixel
  * `output`: the output file path

## Compiling mitsuba3 from source

> This is necessary only when the option `factorize_diffspec` or `DNRFieldRough` is set to `true` in the config file.

This version of mitsuba implements two custom functions: `BSDF::eval_diffuse()` and `BSDF::eval_roughness()`.

To use custom functions for mitsuba3, you need to compile it from [source](mitsuba), and set corresponding environment variables (`PATH/PYTHONPATH`) that are required to run Mitsuba.

Refer to [the document](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html) for building on different platforms. Specifically, on Windows, one can build Mitsuba using:

``` bash
    cd mitsuba
    # add cmake binary to PATH and specify the correct MSVC generator version
    cmake -G "Visual Studio 17 2022" -A x64 -B build
    cmake --build build --config Release
```

Once Mitsuba is built, run the `setpath.sh/.bat/.ps1` script in your build directory to configure environment variables (`PATH/PYTHONPATH`) that are required to run Mitsuba. E.g., on Windows Powershell:

``` bash
    cd mitsuba\build\Release    
    .\setpath.ps1
```