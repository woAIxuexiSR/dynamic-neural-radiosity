import mitsuba as mi
import drjit as dr
import torch
import time

from integrators.lhs_rhs import *

mi.set_variant("cuda_rgb")


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
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        self.model.eval()
        with torch.no_grad():

            ray = mi.Ray3f(ray)

            si, throughput, null_face, _ = first_smooth(scene, sampler, ray, active)
            dr.sync_device()
            L, _, _, valid = render_lhs(scene, self.v, si, self.model)


        return L * throughput, valid & ~null_face, []


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
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        self.model.eval()
        with torch.no_grad():

            ray = mi.Ray3f(ray)

            si, throughput, null_face, _ = first_smooth(scene, sampler, ray, active)
            dr.sync_device()
            L, _, _, _, valid = render_rhs(scene, self.v, sampler, si, self.model)

        return L * throughput, valid & ~null_face, []


class PTIntegrator(mi.SamplingIntegrator):

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        ray = mi.Ray3f(ray)
        si = scene.ray_intersect(ray, active)
        L, valid = render_pt(scene, sampler, si)
        
        return L, valid, []