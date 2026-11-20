import torch
from torch.utils.tensorboard import SummaryWriter

import mitsuba as mi
import drjit as dr
import numpy as np
from tqdm import tqdm
import os
import gc
import json
import argparse
from contextlib import nullcontext

from integrators.lhs_rhs import *
from integrators.integrator import *
from dscene.dscene import DynamicScene
from dscene.sample import *
from model.helper import *
from utils.profiler import TrainingProfiler

CLEANUP_INTERVAL = 500


def train(
    dscene,
    model,
    surface_weight,
    surface_mask,
    config,
    out_dir,
    profile_steps=0,
):
    M = config["rhs_samples"]
    batch_size = config["batch_size"]
    schedule_steps = config["steps"]
    steps = schedule_steps
    if profile_steps > 0:
        steps = min(steps, profile_steps)
    lr = config["learning_rate"]
    loss_interval = config["loss_interval"]
    save_interval = config["save_interval"]
    rhs_update_interval = max(1, schedule_steps // 5)

    indices = dr.arange(mi.UInt, 0, batch_size)
    indices = dr.repeat(indices, M)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step_LR = torch.optim.lr_scheduler.StepLR(
        optimizer, max(1, schedule_steps // 3), 0.33)
    loss_fn = get_loss_fn(config["loss"])
    writer = SummaryWriter()
    tqdm_iter = tqdm(range(steps))
    profiler = TrainingProfiler(steps) if profile_steps > 0 else None

    batch_loss = 0
    warm_up = 100
    v = np.random.uniform(0, 1, dscene.var_num)

    l_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    r_sampler: mi.Sampler = mi.load_dict({"type": "independent"})

    model.train()
    for step in tqdm_iter:
        if profiler is not None:
            profiler.start_step(step)

        if step > warm_up:
            v = np.random.uniform(0, 1, dscene.var_num)

        if profiler is not None:
            update_timer = profiler.measure("scene.update")
        else:
            update_timer = nullcontext()
        with update_timer:
            dscene.update(v)
        scene = dscene.scene

        optimizer.zero_grad()

        l_sampler.seed(step, batch_size)
        r_sampler.seed(step, batch_size * M)

        shape_idx, si_lhs = sample_si(
            scene,
            surface_weight,
            l_sampler.next_1d(),
            l_sampler.next_2d(),
            l_sampler.next_2d(),
        )
        si_rhs = dr.gather(mi.SurfaceInteraction3f, si_lhs, indices)

        if profiler is not None:
            lhs_timer = profiler.measure("render.lhs")
        else:
            lhs_timer = nullcontext()
        with lhs_timer:
            _, Le_lhs, out_lhs, valid_lhs = render_lhs(
                scene, v, si_lhs, model)

        if profiler is not None:
            rhs_timer = profiler.measure("render.rhs")
        else:
            rhs_timer = nullcontext()
        with rhs_timer:
            L_rhs, Le_rhs, out_rhs, weight_rhs, valid_rhs = render_rhs(
                scene, v, r_sampler, si_rhs, model)

        if profiler is not None:
            optimize_timer = profiler.measure("train.optimize")
        else:
            optimize_timer = nullcontext()
        with optimize_timer:
            valid_lhs = valid_lhs.torch().bool()
            out_lhs[~valid_lhs] = 0.0
            lhs = Le_lhs.torch().transpose(0, 1) + out_lhs

            rhs = L_rhs.torch().transpose(0, 1)
            rhs = rhs.reshape(batch_size, M, 3)

            loss = loss_fn(lhs, rhs)
            batch_loss += loss.item()

            loss.backward()
            optimizer.step()
            step_LR.step()

        loss_value = loss.item()
        tqdm_iter.set_description(f"loss: {loss_value:.4f}")

        if step % CLEANUP_INTERVAL == CLEANUP_INTERVAL - 1:
            torch.cuda.empty_cache()
            dr.flush_malloc_cache()
            gc.collect()

        if step % loss_interval == loss_interval - 1:
            writer.add_scalar("loss", batch_loss / loss_interval, step)
            batch_loss = 0

        if step % save_interval == save_interval - 1:
            torch.save(model.state_dict(), out_dir +
                       "/" + config["model_name"])

        if config.get("use_adaptive_rhs", False) and step % rhs_update_interval == rhs_update_interval - 1:
            M = M * 2
            batch_size = batch_size // 2
            indices = dr.arange(mi.UInt, 0, batch_size)
            indices = dr.repeat(indices, M)

    writer.close()
    if profiler is not None:
        profiler.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Radiosity")
    parser.add_argument("-c", type=str, default="config.json")
    parser.add_argument("-m", type=str, default="")
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Run only this many steps with synchronized timing statistics.",
    )
    parser.add_argument(
        "--jit-fusion-encoding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable tiny-cuda-nn JIT fusion for the main encoding.",
    )
    args = parser.parse_args()
    config = json.load(open(args.c, "r"))
    if args.jit_fusion_encoding is not None:
        config.setdefault("model", {})["jit_fusion_encoding"] = (
            args.jit_fusion_encoding
        )
    out_dir = config["output"]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # prepare scene

    scene = mi.load_file(config["scene"], optimize=False)
    mask, m_area = compute_area(scene)
    bbox = scene.bbox()

    dscene = DynamicScene(scene)
    dscene.load_animation(config["animation"])

    # prepare model
    model = get_model(config["model"]["type"], args.m, bbox, config["model"], dscene.var_num)

    # training parameters

    train(
        dscene,
        model,
        m_area,
        mask,
        config["train"],
        out_dir,
        profile_steps=args.profile_steps,
    )
