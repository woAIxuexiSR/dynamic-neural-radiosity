import mitsuba as mi
import drjit as dr
import numpy as np
import json
import argparse
import imgui

import torch

from utils.ui import UI
from model.helper import *
from dscene.dscene import DynamicScene
from integrators.integrator import *

mi.set_variant("cuda_rgb")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dynamic Neural Radiosity")
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")

    args = parser.parse_args()
    config = json.load(open(args.c, "r"))

    # prepare scene

    scene: mi.Scene = mi.load_file(config["scene"])
    bbox = scene.bbox()

    dscene = DynamicScene(scene)
    dscene.load_animation(config["animation"])

    v = np.random.uniform(0, 1, dscene.var_num)
    if config["v"] != "":
        v = np.array(config["v"])
    dscene.update(v)

    # print(dscene.params)

    # prepare model

    model = get_model(config["model"]["type"], args.m,
                       bbox, config["model"], dscene.var_num)
    # rendering

    sensor: mi.Sensor = scene.sensors()[0]
    size = sensor.film().size()

    ui = UI(*size, dscene.camera)

    int_type = 0
    spp = 1
    exposure = 1.0

    path = mi.load_dict({"type": "path", "max_depth": 16})
    LHS = LHSIntegrator(model, config["train"])
    RHS = RHSIntegrator(model, config["train"])

    while not ui.should_close():

        ui.begin_frame()
        dscene.render_ui()

        LHS.v = dscene.v
        RHS.v = dscene.v

        if imgui.tree_node("Render Options", imgui.TREE_NODE_DEFAULT_OPEN):

            _, int_type = imgui.combo("Integrator", int_type, [
                                      "Path", "LHS", "RHS"])
            _, spp = imgui.slider_int("SPP", spp, 1, 4)

            if int_type == 0:
                integrator = path
            elif int_type == 1:
                integrator = LHS
            else:
                integrator = RHS

            _, exposure = imgui.slider_float("Exposure", exposure, 0.1, 5)

            imgui.tree_pop()

        seed = int(ui.duration * 1000)

        img = mi.render(scene, integrator=integrator, spp=spp, seed=seed).torch()

        img = torch.log1p(exposure * img)  # tone mapping
        img = img ** (1 / 2.2)  # gamma correction

        ui.write_texture_gpu(img)
        # ui.write_texture_cpu(img.cpu().numpy())
        ui.end_frame()

    ui.close()
