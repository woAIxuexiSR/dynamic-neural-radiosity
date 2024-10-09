import mitsuba as mi
import drjit as dr
import torch
import numpy as np
import json
import imgui

from dscene.camera import FPSCamera, MovingCamera

mi.set_variant("cuda_rgb")


def val_to_str(value):
    if isinstance(value, mi.Vector3f):
        return "[{:.2f}, {:.2f}, {:.2f}]".format(value.x[0], value.y[0], value.z[0])
    elif isinstance(value, np.ndarray):
        return "[{:.2f}, {:.2f}, {:.2f}]".format(value[0], value[1], value[2])
    return "[{:.2f}]".format(value)


class DynamicScene:
    def __init__(self, scene: mi.Scene):
        self.scene: mi.Scene = scene
        self.params: mi.SceneParameters = mi.traverse(scene)

        self.var_num: int = 0
        self.v: np.ndarray = np.ones(self.var_num) * 0.01

        self.common_vars = {}
        self.translate_vars = {}
        self.rotate_vars = {}
        self.scale_vars = {}
        self.multi_move_vars = {}
        self.is_obj = {}
        self.initial_state = {}

        cam_transform = self.params["PerspectiveCamera.to_world"]
        cam_transform = np.array(cam_transform.matrix).reshape(4, 4)
        bbox = scene.bbox()
        cam_speed = 0.01 * np.array(bbox.max - bbox.min).mean()
        self.camera = FPSCamera(cam_transform, cam_speed)

        self.active_moving_camera = False
        self.moving_Camera = None
        self.camera_v = 0

    def load_animation(self, path):

        with open(path, "r") as f:
            data = json.load(f)

        for t in data["common"]:
            self.add_common_animation(t["param_name"], t["start"], t["end"])
        for t in data["translate"]:
            self.add_translate_animation(
                t["shape_names"], t["start"], t["end"])
        for t in data["rotate"]:
            self.add_rotate_animation(
                t["shape_names"], t["axis"], t["translation"], t["start"], t["end"])
        for t in data["scale"]:
            self.add_scale_animation(t["shape_names"], t["start"], t["end"])
        if "multi-move" in data:
            for t in data["multi-move"]:
                self.add_multi_move_animation(
                    t["shape_names"], t["starts"], t["ends"], init_trans=t.get("initial_transform"))
        
        if "camera" in data:
            cam_config = data["camera"]
            self.active_moving_camera = cam_config["active"]
            self.moving_Camera = MovingCamera(
                cam_config["pos_start"], cam_config["pos_end"], cam_config["rot_start"], cam_config["rot_end"])

        self.v = np.ones(self.var_num) * 0.01

    # helper function for adding initial state
    def add_initial_state(self, is_obj, shape_name):
        if not is_obj:
            return

        if shape_name in self.initial_state:
            return
        self.initial_state[shape_name] = dr.unravel(mi.Point3f,
                                                    self.params[shape_name + ".vertex_positions"])

    # only support vector or scalar, {idx: (param_name, start, end)}
    def add_common_animation(self, param_name, start, end):
        if isinstance(start, list):
            start, end = mi.Vector3f(start), mi.Vector3f(end)
        self.common_vars[self.var_num] = (param_name, start, end)
        self.var_num += 1

    # vector, {idx: (shape_names, start, end)}
    def add_translate_animation(self, shape_names, start, end):
        start, end = mi.Vector3f(start), mi.Vector3f(end)
        for shape_name in shape_names:
            _is_obj = (self.params.get(shape_name + ".to_world") is None)
            self.is_obj[shape_name] = _is_obj
            self.add_initial_state(_is_obj, shape_name)
        self.translate_vars[self.var_num] = (shape_names, start, end)
        self.var_num += 1

    # in degree, {idx: (shape_names, axis, translation, start, end)}
    def add_rotate_animation(self, shape_names, axis, translation, start, end):
        axis, translation = mi.Vector3f(axis), mi.Vector3f(translation)
        for shape_name in shape_names:
            _is_obj = (self.params.get(shape_name + ".to_world") is None)
            self.is_obj[shape_name] = _is_obj
            self.add_initial_state(_is_obj, shape_name)
        self.rotate_vars[self.var_num] = (
            shape_names, axis, translation, start, end)
        self.var_num += 1

    # scalar, {idx: (shape_names, start, end)}
    def add_scale_animation(self, shape_names, start, end):
        for shape_name in shape_names:
            _is_obj = (self.params.get(shape_name + ".to_world") is None)
            self.is_obj[shape_name] = _is_obj
            self.add_initial_state(_is_obj, shape_name)
        self.scale_vars[self.var_num] = (shape_names, start, end)
        self.var_num += 1
    
    # vector, {idx: (shape_names, start, end)}
    def add_multi_move_animation(self, shape_names, starts, ends, init_trans=None):
        starts, ends = torch.Tensor(starts).to(device='cuda'), torch.Tensor(ends).to(device='cuda')
        starts, ends = starts.reshape(-1, 3), ends.reshape(-1, 3)
        if init_trans is not None:
            T = np.array(init_trans).reshape(4, 4)
            T = mi.Transform4f(T)
        for shape_name in shape_names:
            _is_obj = (self.params.get(shape_name + ".to_world") is None)
            self.is_obj[shape_name] = _is_obj
            if _is_obj:
                self.params[shape_name + ".vertex_positions"] = dr.ravel(T @
                    dr.unravel(mi.Point3f, self.params[shape_name + ".vertex_positions"])
                )
            self.add_initial_state(_is_obj, shape_name)
        self.multi_move_vars[self.var_num] = (shape_names, starts, ends)
        self.var_num += 1

    def apply_obj_transformation(self, shape_name, T):
        self.params[shape_name +
                    ".vertex_positions"] = dr.ravel(T @ self.initial_state[shape_name])

    # v : list of [0, 1] that represents the progress of each animation
    def update(self, v: np.ndarray, changed=True):

        # update common states
        common_states = {}
        for idx, (param_name, start, end) in self.common_vars.items():
            if not changed: continue
            c = dr.lerp(start, end, v[idx])
            if common_states.get(param_name) is None:
                common_states[param_name] = c
            else:
                common_states[param_name] += c
        for param_name, c in common_states.items():
            self.params[param_name] = c

        # update translate
        translate_states = {}
        for idx, (shape_names, start, end) in self.translate_vars.items():
            if not changed: continue
            c = dr.lerp(start, end, v[idx])
            for shape_name in shape_names:
                if translate_states.get(shape_name) is None:
                    translate_states[shape_name] = c
                else:
                    translate_states[shape_name] += c
        for shape_name, c in translate_states.items():
            if self.is_obj[shape_name]:
                T = mi.Transform4f.translate(c)
                self.apply_obj_transformation(shape_name, T)
            else:
                mat = self.params[shape_name + ".to_world"].matrix
                mat[3] = mi.Vector4f(c.x, c.y, c.z, 1)
                self.params[shape_name + ".to_world"] = mi.Transform4f(mat)

        # update rotate
        for idx, (shape_names, axis, translation, start, end) in self.rotate_vars.items():
            if not changed: continue
            c = dr.lerp(start, end, v[idx])
            T = mi.Transform4f.translate(
                translation) @ mi.Transform4f.rotate(axis, c) @ mi.Transform4f.translate(-translation)
            for shape_name in shape_names:
                if self.is_obj[shape_name]:
                    self.apply_obj_transformation(shape_name, T)
                else:
                    self.params[shape_name + ".to_world"] = T

        # update scale
        for idx, (shape_names, start, end) in self.scale_vars.items():
            if not changed: continue
            c = dr.lerp(start, end, v[idx])
            for shape_name in shape_names:
                if self.is_obj[shape_name]:
                    T = mi.Transform4f.scale(c)
                    self.apply_obj_transformation(shape_name, T)
                else:
                    mat = self.params[shape_name + ".to_world"]
                    translate = mat.translation()
                    
                    res = mi.Transform4f.scale(c)
                    res = mi.Matrix4f(res.matrix)
                    res[3] = mi.Vector4f(
                        translate.x, translate.y, translate.z, 1.0)
                    self.params[shape_name + ".to_world"] = mi.Transform4f(res)
                    
        # update multi-move
        multi_move_states = {}
        for idx, (shape_names, starts, ends) in self.multi_move_vars.items():
            if not changed: continue
            for i, shape_name in enumerate(shape_names):
                start = mi.Vector3f(starts[i].unsqueeze(0))
                end = mi.Vector3f(ends[i].unsqueeze(0))
                c = dr.lerp(start, end, v[idx])
                if multi_move_states.get(shape_name) is None:
                    multi_move_states[shape_name] = c
                else:
                    multi_move_states[shape_name] += c
        for shape_name, c in multi_move_states.items():
            if self.is_obj[shape_name]:
                T = mi.Transform4f.translate(c)
                self.apply_obj_transformation(shape_name, T)
            else:
                mat = self.params[shape_name + ".to_world"].matrix
                mat[3] = mi.Vector4f(c.x, c.y, c.z, 1)
                self.params[shape_name + ".to_world"] = mi.Transform4f(mat)

        # update camera
        if self.active_moving_camera:
            self.params["PerspectiveCamera.to_world"] = self.moving_Camera.get_transform(
                self.camera_v)

        self.params.update()
        self.v = v

    def render_ui(self):

        changed = False

        if imgui.tree_node("Camera Control", imgui.TREE_NODE_DEFAULT_OPEN):

            fov = self.params["PerspectiveCamera.x_fov"]
            _, fov = imgui.slider_float("fov", fov[0], 10, 90, "%.1f")
            self.params["PerspectiveCamera.x_fov"] = fov

            if self.active_moving_camera:
                _, v = imgui.slider_float(
                    "camera v", self.camera_v, 0, 1, "%.2f")
                self.camera_v = v
                self.params["PerspectiveCamera.to_world"] = self.moving_Camera.get_transform(
                    self.camera_v)
            else:
                self.params["PerspectiveCamera.to_world"] = self.camera.get_transform()

                imgui.text("pos: " + val_to_str(self.camera.pos))
                imgui.text("front: " + val_to_str(self.camera.z))
                imgui.text("up: " + val_to_str(self.camera.y))
                imgui.text("right: " + val_to_str(self.camera.x))

            imgui.tree_pop()

        if imgui.tree_node("Variable Control", imgui.TREE_NODE_DEFAULT_OPEN):

            for idx, (param_name, start, end) in self.common_vars.items():
                c = dr.lerp(start, end, self.v[idx])
                imgui.text(param_name + " : " + val_to_str(c))
                _c, v = imgui.slider_float(
                    "variable " + str(idx), self.v[idx], 0.01, 0.99, "%.2f")
                changed |= _c
                self.v[idx] = v

            for idx, (shape_names, start, end) in self.translate_vars.items():
                c = dr.lerp(start, end, self.v[idx])
                param_name = str(shape_names) + " translate"
                imgui.text(param_name + " : " + val_to_str(c))
                _c, v = imgui.slider_float(
                    "variable " + str(idx), self.v[idx], 0.01, 0.99, "%.2f")
                changed |= _c
                self.v[idx] = v

            for idx, (shape_names, _, _, start, end) in self.rotate_vars.items():
                c = dr.lerp(start, end, self.v[idx])
                param_name = str(shape_names) + " rotate"
                imgui.text(param_name + " : " + val_to_str(c))
                _c, v = imgui.slider_float(
                    "variable " + str(idx), self.v[idx], 0.01, 0.99, "%.2f")
                changed |= _c
                self.v[idx] = v

            for idx, (shape_names, start, end) in self.scale_vars.items():
                c = dr.lerp(start, end, self.v[idx])
                param_name = str(shape_names) + " scale"
                imgui.text(param_name + " : " + val_to_str(c))
                _c, v = imgui.slider_float(
                    "variable " + str(idx), self.v[idx], 0.01, 0.99, "%.2f")
                changed |= _c
                self.v[idx] = v
            
            for idx, (shape_names, starts, ends) in self.multi_move_vars.items():
                start = mi.Vector3f(starts[0].unsqueeze(0))
                end = mi.Vector3f(ends[0].unsqueeze(0))
                c = dr.lerp(start, end, self.v[idx])
                param_name = str(shape_names) + " multi-move"
                imgui.text(param_name + " : " + val_to_str(c))
                _c, v = imgui.slider_float(
                    "variable " + str(idx), self.v[idx], 0.01, 0.99, "%.2f")
                changed |= _c
                self.v[idx] = v

            imgui.tree_pop()

        self.update(self.v, changed)
