# Scenes and Animations

## Scene Structure

Each scene is stored in its own subdirectory under `scenes/`:

```
scenes/<scene-name>/
├── scene.xml          # Mitsuba 3 scene description
├── animation.json     # Animation definition
├── textures/          # Texture files
├── models/            # Mesh files
└── ref.exr            # Reference image (optional)
```

## Animation Format

Animations are defined in a JSON file with the following fields:

### `common`

Interpolate arbitrary scene parameters (e.g., material properties, light intensity) between start and end values.

```json
"common": [
    {
        "param_name": "WallsBSDF.reflectance.value",
        "start": [0.2, 0.2, 0.2],
        "end": [0.5, 0.05, 0.3]
    }
]
```

### `translate`

Linearly translate a group of shapes.

```json
"translate": [
    {
        "shape_names": ["Ball"],
        "start": [0.2, 1.0, 0],
        "end": [-0.4, 0.4, 0]
    }
]
```

### `rotate`

Rotate a group of shapes around an axis with a pivot point.

```json
"rotate": [
    {
        "shape_names": ["EnvironmentMapEmitter"],
        "axis": [0, 1, 0],
        "translation": [0, 0, 0],
        "start": 90,
        "end": 120
    }
]
```

### `scale`

Scale a group of shapes between start and end values.

```json
"scale": [
    {
        "shape_names": ["Ball"],
        "start": 0.4,
        "end": 0.2
    }
]
```

### `camera` (optional)

Define a moving camera path with position and rotation (quaternion XYZW) interpolation.

```json
"camera": {
    "active": true,
    "pos_start": [-0.25, 1.8, 7],
    "pos_end": [0.5, 1.8, 5],
    "rot_start": [0, 0.999, -0.015, -0.034],
    "rot_end": [0, 0.998, -0.02, -0.05]
}
```

### Constraints

- For OBJ shapes, only one transformation type is supported per shape.
- For other shapes, translation and scale can be composed, but rotation cannot be composed with other transformations.
- All shapes within a single animation entry must be of the same type (all OBJ or all non-OBJ).
- Each animation entry is controlled by an independent variable `v[i] ∈ [0, 1]`.
