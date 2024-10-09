import mitsuba as mi
import drjit as dr
import numpy as np
import json
import argparse

from dscene.dscene import DynamicScene
from integrators.integrator import *
from model.helper import *

mi.set_variant("cuda_rgb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dynamic Neural Radiosity")
    parser.add_argument("-t", type=str, default="path")
    parser.add_argument("-s", type=int, default=1)
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")
    parser.add_argument("-o", type=str, default="hello.exr")

    args = parser.parse_args()
    config = json.load(open(args.c, "r"))

    # prepare scene

    scene: mi.Scene = mi.load_file(config["scene"])
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
        
        model = get_model(config["model"]["type"], args.m, bbox, config["model"], dscene.var_num)
        
        if args.t == "LHS":
            integrator = LHSIntegrator(model)
            integrator.v = v
        elif args.t == "RHS":
            integrator = RHSIntegrator(model)
            integrator.v = v
            
    else:
        raise NotImplementedError

    dscene.update(v)
    
    size = dscene.scene.sensors()[0].film().size()
    img: mi.TensorXf = dr.zeros(mi.TensorXf, (size[1], size[0], 3))
    
    max_spp_per_iter = 1024 if args.t == "path" else 1
    iter_num = (args.s + max_spp_per_iter - 1) // max_spp_per_iter
    spp = args.s if iter_num == 1 else max_spp_per_iter
    
    # start = time.time()
    # for j in range(10):
    for i in range(iter_num):
        start = time.time()

        img += mi.render(dscene.scene, integrator=integrator, spp=spp, seed=i + 2)
        dr.flush_malloc_cache()

        torch.cuda.synchronize()
        print("Time: ", time.time() - start)
    img = img / iter_num

    # print("Time: ", time.time() - start)
    
    img = mi.Bitmap(img)
    img.write(args.o)
