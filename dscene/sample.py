import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("cuda_rgb")


def compute_area(scene: mi.Scene):
    m_area = []

    for shape in scene.shapes():
        if not shape.is_emitter() and mi.has_flag(
            shape.bsdf().flags(), mi.BSDFFlags.Smooth
        ):
            m_area.append(shape.surface_area())
        else:
            m_area.append([0])
    m_area = np.array(m_area)[:, 0]
    m_area = m_area / m_area.sum()
    mask = (m_area > 0)

    return mask, m_area


def sample_si(
    scene: mi.Scene,
    m_area,
    sample1: mi.Float,
    sample2: mi.Point2f,
    sample3: mi.Point2f,
    active=True,
) -> mi.SurfaceInteraction3f:
    sampler = mi.DiscreteDistribution(m_area)
    shape_idx = sampler.sample(sample1, active)
    shapes: mi.Shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx, active)

    ps = shapes.sample_position(0, sample2, active)
    si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
    si.shape = shapes
    si.t = mi.Float(0)

    active_two_sided = mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide)
    si.wi = dr.select(
        active_two_sided,
        mi.warp.square_to_uniform_sphere(sample3),
        mi.warp.square_to_uniform_hemisphere(sample3),
    )

    return shape_idx, si
