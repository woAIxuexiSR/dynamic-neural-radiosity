import torch

import mitsuba as mi
import drjit as dr
import numpy as np
import json
import argparse
import os
from contextlib import nullcontext

from dscene.dscene import DynamicScene
from integrators.integrator import *
from utils.profiler import RenderProfiler

mi.set_variant(os.environ.get("MITSUBA_VARIANT", "cuda_rgb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dynamic Neural Radiosity")
    parser.add_argument("-t", type=str, default="path")
    parser.add_argument("-s", type=int, default=1)
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")
    parser.add_argument("-o", type=str, default="hello.exr")
    parser.add_argument(
        "--jit-fusion-encoding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable tiny-cuda-nn JIT fusion for the main encoding.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print synchronized timing for setup, render passes, and output.",
    )

    args = parser.parse_args()
    config = json.load(open(args.c, "r"))
    if args.jit_fusion_encoding is not None:
        config.setdefault("model", {})["jit_fusion_encoding"] = (
            args.jit_fusion_encoding
        )
    profiler = None

    # prepare scene

    if args.profile:
        profiler = RenderProfiler(0)

    timer = profiler.measure if profiler is not None else lambda _: nullcontext()

    with timer("scene.load"):
        scene: mi.Scene = mi.load_file(config["scene"], optimize=False)
    bbox = scene.bbox()

    dscene = DynamicScene(scene)
    dscene.load_animation(config["animation"])

    # rendering

    v = np.random.uniform(0, 1, dscene.var_num)
    if config["v"] != "":
        v = np.array(config["v"])

    if dscene.active_moving_camera:
        cam_v = np.random.uniform(0, 1)
        if config["cam_v"] != "":
            cam_v = config["cam_v"]
        dscene.camera_v = cam_v

    integrator = mi.load_dict({"type": "path", "max_depth": 16})
    if args.t == "path":
        pass
    elif args.t == "LHS" or args.t == "RHS":
        
        # prepare model
        
        from model.helper import get_model

        with timer("model.load"):
            model = get_model(
                config["model"]["type"],
                args.m,
                bbox,
                config["model"],
                dscene.var_num,
            )
        
        if args.t == "LHS":
            integrator = LHSIntegrator(model, profiler=profiler)
            integrator.v = v
        elif args.t == "RHS":
            integrator = RHSIntegrator(model, profiler=profiler)
            integrator.v = v
            
    else:
        raise NotImplementedError

    dscene.update(v)
    
    size = dscene.scene.sensors()[0].film().size()
    sensor = dscene.scene.sensors()[0]
    img: mi.TensorXf = dr.zeros(mi.TensorXf, (size[1], size[0], 3))
    
    max_spp_per_iter = 1024 if args.t == "path" else 1
    iter_num = (args.s + max_spp_per_iter - 1) // max_spp_per_iter
    spp = args.s if iter_num == 1 else max_spp_per_iter
    if profiler is not None:
        profiler.passes = iter_num
    
    for i in range(iter_num):
        pass_scope = profiler.pass_scope() if profiler is not None else nullcontext()
        with pass_scope:
            with timer("render.pass"):
                rendered = mi.render(
                    scene=dscene.scene,
                    sensor=sensor,
                    integrator=integrator,
                    spp=spp,
                    seed=i,
                )
            img += rendered
            dr.flush_malloc_cache()

    img = img / iter_num
    
    img = mi.Bitmap(img)
    with timer("image.write"):
        img.write(args.o)

    if profiler is not None:
        profiler.report()
