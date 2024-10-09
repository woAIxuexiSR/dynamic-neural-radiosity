import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import gc
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tinycudann as tcnn

from integrators.lhs_rhs import *
from integrators.integrator import *
from dscene.dscene import DynamicScene
from dscene.sample import *
from model.helper import *

            
def train(dscene, model, surface_weight, surface_mask, config, out_dir):
    M = config["rhs_samples"]
    batch_size = config["batch_size"]
    steps = config["steps"]
    lr = config["learning_rate"]
    save_interval = config["save_interval"]
    rhs_update_interval = steps // 5

    indices = dr.arange(mi.UInt, 0, batch_size)
    indices = dr.repeat(indices, M)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step_LR = torch.optim.lr_scheduler.StepLR(optimizer, steps // 3, 0.33)
    loss_fn = get_loss_fn(config["loss"])
    writer = SummaryWriter()
    tqdm_iter = tqdm(range(steps))

    batch_loss = 0
    warm_up = 100
    v = np.random.uniform(0, 1, dscene.var_num)

    model.train()
    for step in tqdm_iter:

        if step > warm_up:
            v = np.random.uniform(0, 1, dscene.var_num)

        dscene.update(v)
        scene = dscene.scene

        optimizer.zero_grad()

        l_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        r_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        l_sampler.seed(step, batch_size)
        r_sampler.seed(step, batch_size * M)

        shape_idx, si_lhs = sample_si(
            scene, surface_weight, l_sampler.next_1d(), l_sampler.next_2d(), l_sampler.next_2d()
        )
        si_rhs = dr.gather(mi.SurfaceInteraction3f, si_lhs, indices)

        _, Le_lhs, out_lhs, valid_lhs = render_lhs(scene, v, si_lhs, model)
        L_rhs, Le_rhs, out_rhs, weight_rhs, valid_rhs = render_rhs(
            scene, v, r_sampler, si_rhs, model)

        valid_lhs = valid_lhs.torch().bool()
        out_lhs[~valid_lhs] = 0.0
        lhs = Le_lhs.torch() + out_lhs

        rhs = L_rhs.torch()
        rhs = rhs.reshape(batch_size, M, 3)

        loss = loss_fn(lhs, rhs)
        batch_loss += loss.item()

        loss.backward()
        optimizer.step()
        step_LR.step()

        tqdm_iter.set_description(f"loss: {loss.item():.4f}")
        torch.cuda.empty_cache()
        dr.flush_malloc_cache()
        gc.collect()

        if step % save_interval == save_interval - 1:
            writer.add_scalar("loss", batch_loss / save_interval, step)
            batch_loss = 0
            torch.save(model.state_dict(), out_dir +
                       "/" + config["model_name"])

        if config.get("use_adaptive_rhs", False) and step % rhs_update_interval == rhs_update_interval - 1:
            M = M * 2
            batch_size = batch_size // 2
            indices = dr.arange(mi.UInt, 0, batch_size)
            indices = dr.repeat(indices, M)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Radiosity")
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")

    args = parser.parse_args()
    config = json.load(open(args.c, "r"))

    out_dir = config["output"]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # prepare scene

    scene = mi.load_file(config["scene"])
    mask, m_area = compute_area(scene)
    bbox = scene.bbox()

    dscene = DynamicScene(scene)
    dscene.load_animation(config["animation"])

    # prepare model
    model = get_model(config["model"]["type"], args.m, bbox, config["model"], dscene.var_num)

    # training parameters

    train(dscene, model, m_area, mask, config["train"], out_dir)
