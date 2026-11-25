import torch

import mitsuba as mi
import drjit as dr
import time
import os

from integrators.lhs_rhs import *

mi.set_variant(os.environ.get("MITSUBA_VARIANT", "cuda_rgb"))


def sync_drjit():
    sync = getattr(dr, "sync_device", None)
    if sync is None:
        sync = dr.sync_thread
    sync()


class LHSIntegrator(mi.SamplingIntegrator):

    def __init__(self, model, config: dict={}):
        super().__init__(mi.Properties())
        self.model = model
        self.v = None
        self.config = config

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, mi.Bool]:

        self.model.eval()
        with torch.no_grad():

            ray = mi.Ray3f(ray)

            si, throughput, null_face, _ = first_smooth(scene, sampler, ray, active)
            sync_drjit()
            L, _, _, valid = render_lhs(
                scene, self.v, si, self.model, use_mitsuba_model=True)

        return L * throughput, valid & ~null_face


class RHSIntegrator(mi.SamplingIntegrator):

    def __init__(self, model, config: dict={}):
        super().__init__(mi.Properties())
        self.model = model
        self.v = None
        self.config = config

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, mi.Bool]:

        self.model.eval()
        with torch.no_grad():

            ray = mi.Ray3f(ray)

            si, throughput, null_face, _ = first_smooth(scene, sampler, ray, active)
            sync_drjit()
            L, _, _, _, valid = render_rhs(
                scene, self.v, sampler, si, self.model, use_mitsuba_model=True)

        return L * throughput, valid & ~null_face


class PTIntegrator(mi.SamplingIntegrator):

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, mi.Bool]:

        ray = mi.Ray3f(ray)
        si = scene.ray_intersect(ray, active_mask(active))
        L, valid = render_pt(scene, sampler, si)
        
        return L, valid
