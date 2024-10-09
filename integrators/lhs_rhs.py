import mitsuba as mi
import drjit as dr
import numpy as np
import torch

mi.set_variant("cuda_rgb")


def first_smooth(
    scene: mi.Scene,
    sampler: mi.Sampler,
    ray: mi.Ray3f,
    active: bool = True
) -> tuple[mi.SurfaceInteraction3f, mi.Color3f, bool]:

    with dr.suspend_grad():

        ray = mi.Ray3f(ray)
        final_si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        active = mi.Bool(active)
        throughput = mi.Color3f(1.0)
        spec_mask = mi.Bool(False)
        depth = mi.UInt32(0)
        null_face = mi.Bool(True)

        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "first smooth surface",
            lambda: (
                sampler,
                ray,
                final_si,
                active,
                throughput,
                spec_mask,
                depth,
                null_face
            )
        )

        max_depth = 16
        loop.set_max_iterations(max_depth)

        while loop(active):

            si = scene.ray_intersect(ray, active)
            bsdf: mi.BSDF = si.bsdf(ray)
            final_si[active] = si

            spec_mask |= mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta) & mi.has_flag(
                bsdf.flags(), mi.BSDFFlags.BackSide)

            # null_face &= mi.has_flag(bsdf.flags(), mi.BSDFFlags.Null) | (
            #     ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (
            #         si.wi.z < 0)
            # )
            null_face &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (si.wi.z < 0)

            active &= si.is_valid() & ~null_face & (depth < max_depth)
            opacity = bsdf.eval_null_transmission(si) > mi.Color3f(0.98)
            active &= (~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)) | (
                opacity.x & opacity.y & opacity.z
            )

            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            throughput[active] *= bsdf_weight
            depth[si.is_valid()] += 1

    return final_si, throughput, null_face, spec_mask


def extract_input(si: mi.SurfaceInteraction3f):
    pos = si.p
    dirs = si.to_world(si.wi)
    normals = si.sh_frame.n
    normals = dr.select(dr.dot(dirs, normals) < 0, -normals, normals)

    si_view = mi.SurfaceInteraction3f(si)
    si_view.wi = mi.Point3f([0.353553, 0.353553, 0.866025])
    albedo = si_view.bsdf().eval_diffuse_reflectance(si_view)

    return pos, dirs, normals, albedo


def render_lhs(scene: mi.Scene, v: np.ndarray, si: mi.SurfaceInteraction3f, model):


    with dr.suspend_grad():

        Le = si.emitter(scene).eval(si)

        valid = si.is_valid() & ~(~mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide) & (
            si.wi.z < 0
        ))


        pos, dirs, normals, albedo = extract_input(si)
        out = model(pos, dirs, normals, albedo, v)

        not_emitter = dr.eq(Le, mi.Color3f(0.0))
        active_nr = valid & not_emitter.x & not_emitter.y & not_emitter.z

        L = Le + dr.select(active_nr, mi.Color3f(out), mi.Color3f(0.0))

    # mi.Color3f, mi.Color3f, torch.Tensor, mi.Bool
    return L, Le, out, active_nr


def render_rhs(scene: mi.Scene, v: np.ndarray, sampler: mi.Sampler, si: mi.SurfaceInteraction3f, model):

    si: mi.SurfaceInteraction3f = mi.SurfaceInteraction3f(si)
    ray: mi.Ray3f = mi.Ray3f(si.p, si.to_world(si.wi))
    active = mi.Bool(True)
    throughput = mi.Color3f(1.0)
    result = mi.Color3f(0.0)

    valid_ray = mi.Bool(scene.environment() is not None)

    # Variables caching information from the previous bounce
    prev_si: mi.Interaction3f = dr.zeros(mi.Interaction3f)
    prev_bsdf_pdf = mi.Float(1.0)
    prev_bsdf_delta = mi.Bool(True)
    bsdf_ctx = mi.BSDFContext()

    # while(loop):

    # ---------------------- Direct emission ----------------------

    mis_weight = mi.Float(1.0)

    mis_weight[~prev_bsdf_delta] = 0.0

    ds: mi.DirectionSample3f = mi.DirectionSample3f(
        scene, si, prev_si)
    em_pdf = mi.Float(0.0)

    em_pdf = scene.pdf_emitter_direction(
        prev_si, ds, ~prev_bsdf_delta)

    mis_bsdf = dr.detach(
        dr.select(prev_bsdf_pdf > 0, prev_bsdf_pdf / (prev_bsdf_pdf + em_pdf), 0))
    mis_weight[~prev_bsdf_delta] = mis_bsdf

    result = dr.fma(
        throughput,
        si.emitter(scene).eval(si) * mis_weight,
        result
    )

    active_next = si.is_valid()

    bsdf: mi.BSDF = si.bsdf(ray)

    # ---------------------- Emitter sampling ----------------------

    active_em = active_next & mi.has_flag(
        bsdf.flags(), mi.BSDFFlags.Smooth)

    ds, em_weight = scene.sample_emitter_direction(
        si, sampler.next_2d(), True, active_em
    )

    wo = si.to_local(ds.d)

    # ------ Evaluate BSDF * cos(theta) and sample direction -------

    sample1 = sampler.next_1d()
    sample2 = sampler.next_2d()

    bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
        bsdf_ctx, si, wo, sample1, sample2
    )

    # --------------- Emitter sampling contribution ----------------

    mi_em = mi.Float(1.0)
    mi_em = dr.select(ds.delta, 1.0, dr.detach(
        dr.select((ds.pdf > 0) & (bsdf_pdf > 0), ds.pdf / (ds.pdf + bsdf_pdf), 0)))

    result[active_em] = dr.fma(
        throughput, bsdf_val * em_weight * mi_em, result)

    # ---------------------- BSDF sampling ----------------------

    ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

    # ------ Update loop variables based on current interaction ------

    throughput *= bsdf_weight
    valid_ray |= (
        active
        & si.is_valid()
        & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
    )

    prev_si = mi.Interaction3f(si)
    prev_bsdf_pdf = bsdf_sample.pdf
    prev_bsdf_delta = mi.has_flag(
        bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

    # -------------------- Stopping criterion ---------------------

    active = active_next

    si = scene.ray_intersect(ray, active)

    mis_weight = mi.Float(1.0)

    mis_weight[~prev_bsdf_delta] = 0.0

    ds: mi.DirectionSample3f = mi.DirectionSample3f(
        scene, si, prev_si)
    em_pdf = mi.Float(0.0)

    em_pdf = scene.pdf_emitter_direction(
        prev_si, ds, ~prev_bsdf_delta)

    mis_bsdf = dr.detach(
        dr.select(prev_bsdf_pdf > 0, prev_bsdf_pdf / (prev_bsdf_pdf + em_pdf), 0))
    mis_weight[~prev_bsdf_delta] = mis_bsdf

    si, throughput2, null_face, _ = first_smooth(scene, sampler, ray, active)
    throughput *= throughput2

    result = dr.fma(
        throughput,
        si.emitter(scene).eval(si) * mis_weight,
        result
    )

    not_emitter = dr.eq(si.emitter(scene).eval(si), mi.Color3f(0.0))
    active_nr = active & ~null_face & si.is_valid(
    ) & not_emitter.x & not_emitter.y & not_emitter.z

    pos, dirs, normals, albedo = extract_input(si)
    out = model(pos, dirs, normals, albedo, v)

    Le = mi.Color3f(result)
    result = dr.fma(
        throughput,
        dr.select(active_nr, mi.Color3f(out), mi.Color3f(0.0)),
        result
    )

    return result, Le, out, throughput, active_nr


def render_pt(scene: mi.Scene, sampler: mi.Sampler, si: mi.SurfaceInteraction3f):

    si: mi.SurfaceInteraction3f = mi.SurfaceInteraction3f(si)
    ray: mi.Ray3f = mi.Ray3f(si.p, si.to_world(si.wi))
    active = mi.Bool(True)
    throughput = mi.Color3f(1.0)
    result = mi.Color3f(0.0)
    depth = mi.UInt32(0)

    valid_ray = mi.Bool(scene.environment() is not None)

    # Variables caching information from the previous bounce
    prev_si: mi.Interaction3f = dr.zeros(mi.Interaction3f)
    prev_bsdf_pdf = mi.Float(1.0)
    prev_bsdf_delta = mi.Bool(True)
    bsdf_ctx = mi.BSDFContext()

    loop = mi.Loop(
        "Path Tracer",
        lambda: (
            sampler,
            ray,
            si,
            throughput,
            result,
            depth,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            active,
        )
    )

    max_depth = 8
    loop.set_max_iterations(max_depth)

    while loop(active):

        # si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        # ---------------------- Direct emission ----------------------

        mis_weight = mi.Float(1.0)

        mis_weight[~prev_bsdf_delta] = 0.0

        ds: mi.DirectionSample3f = mi.DirectionSample3f(
            scene, si, prev_si)
        em_pdf = mi.Float(0.0)

        em_pdf = scene.pdf_emitter_direction(
            prev_si, ds, ~prev_bsdf_delta)

        mis_bsdf = dr.detach(
            dr.select(prev_bsdf_pdf > 0, prev_bsdf_pdf / (prev_bsdf_pdf + em_pdf), 0))
        mis_weight[~prev_bsdf_delta] = mis_bsdf

        result = dr.fma(
            throughput,
            si.emitter(scene).eval(si) * mis_weight,
            result
        )

        active_next = ((depth + 1) < max_depth) & si.is_valid()

        bsdf: mi.BSDF = si.bsdf(ray)

        # ---------------------- Emitter sampling ----------------------

        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )

        wo = si.to_local(ds.d)

        # ------ Evaluate BSDF * cos(theta) and sample direction -------

        sample1 = sampler.next_1d()
        sample2 = sampler.next_2d()

        bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
            bsdf_ctx, si, wo, sample1, sample2
        )

        # --------------- Emitter sampling contribution ----------------

        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

        mi_em = mi.Float(1.0)
        mi_em = dr.select(ds.delta, 1.0, dr.detach(
            dr.select(ds.pdf > 0, ds.pdf / (ds.pdf + bsdf_pdf), 0)))

        result[active_em] = dr.fma(
            throughput, bsdf_val * em_weight * mi_em, result)

        # ---------------------- BSDF sampling ----------------------

        bsdf_weight = si.to_world_mueller(
            bsdf_weight, -bsdf_sample.wo, si.wi)

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        valid_ray |= (
            active
            & si.is_valid()
            & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
        )

        prev_si = mi.Interaction3f(si)
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # -------------------- Stopping criterion ---------------------

        depth[si.is_valid()] += 1

        throughput_max = dr.max(throughput)

        rr_prop = dr.minimum(throughput_max, 0.95)
        rr_active = (depth >= 3)
        rr_continue = (sampler.next_1d() < rr_prop)

        throughput[rr_active] *= dr.rcp(rr_prop)

        active = (
            active_next & (~rr_active | rr_continue) & (
                dr.neq(throughput_max, 0.0))
        )

        si = scene.ray_intersect(ray, active_next)

    return dr.select(valid_ray, result, 0.0), valid_ray
