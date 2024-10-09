import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
from imgui.integrations.glfw import GlfwRenderer

import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl

import numpy as np
import drjit as dr
import torch
import time
import os
import warnings


class UI:

    def __init__(self, width, height, camera, name="Render"):

        self.width = width
        self.height = height
        self.name = name

        self.camera = camera
        self.first_mouse = True
        self.prev_x = 0
        self.prev_y = 0
        
        self.current = time.time()
        self.duration = 0

        # initialize glfw
        if not glfw.init():
            print("Failed to initialize GLFW")
            exit()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(width, height, name, None, None)

        if not window:
            print("Failed to create window")
            glfw.terminate()
            exit()

        glfw.make_context_current(window)

        glViewport(0, 0, width, height)

        glfw.swap_interval(0)
        self.window = window

        # initialize imgui
        imgui.create_context()
        self.impl = GlfwRenderer(window)

        # create shader, vao, texture
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.program = self.create_program(
            file_path + "/shader/hello.vert",
            file_path + "/shader/hello.frag"
        )
        self.vao = self.create_vao()
        self.texture = self.create_texture()

        import pycuda.gl.autoinit
        self.pbo = self.create_pbo(width, height)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.bufobj = cuda_gl.BufferObject(int(self.pbo))

    def close(self):

        self.bufobj.unregister()

        glDeleteProgram(self.program)
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.pbo])
        glDeleteTextures(1, [self.texture])

        self.impl.shutdown()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def create_program(self, vertex_path, fragment_path):

        with open(vertex_path, "r") as f:
            vertex_shader = f.read()

        with open(fragment_path, "r") as f:
            fragment_shader = f.read()

        vertex = OpenGL.GL.shaders.compileShader(
            vertex_shader, GL_VERTEX_SHADER)
        fragment = OpenGL.GL.shaders.compileShader(
            fragment_shader, GL_FRAGMENT_SHADER)

        return OpenGL.GL.shaders.compileProgram(vertex, fragment)

    def create_vao(self):

        # flip vertically
        quad = np.array([
            # position 2, texcoord 2
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
            1.0, -1.0,  1.0, 1.0,

            -1.0,  1.0,  0.0, 0.0,
            1.0, -1.0,  1.0, 1.0,
            1.0,  1.0,  1.0, 0.0
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 *
                              quad.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 *
                              quad.itemsize, ctypes.c_void_p(2 * quad.itemsize))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glDeleteBuffers(1, [vbo])

        return vao

    def create_pbo(self, w, h):

        data = np.zeros((w * h * 3), dtype=np.float32)
        pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, pbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return pbo

    def create_texture(self):

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, self.width,
                     self.height, 0, GL_RGB, GL_FLOAT, None)

        return texture

    def process_input(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.move(1, 1)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.move(1, -1)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.move(2, 1)
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.move(2, -1)
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.camera.move(0, 1)
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.camera.move(0, -1)
            
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(self.window)
            if self.first_mouse:
                self.first_mouse = False
                self.prev_x = xpos
                self.prev_y = ypos
            speed = 0.04
            xoffset = (xpos - self.prev_x) * speed
            yoffset = (ypos - self.prev_y) * speed
            self.camera.rotate(xoffset, yoffset)
        else:
            self.first_mouse = True

    def should_close(self):
        return glfw.window_should_close(self.window)

    def begin_frame(self):

        t = time.time()
        fps = 1.0 / (t - self.current)
        self.duration += t - self.current
        self.current = t

        imgui.new_frame()

        imgui.begin("Options")

        imgui.text("Time: {:.1f}".format(self.duration))
        imgui.text("FPS: {:.1f}".format(fps))

    # img: (height, width, 3) np.float32
    def write_texture_cpu(self, img):

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width,
                        self.height, GL_RGB, GL_FLOAT, img)

    # img: (height, width, 3) torch.float32
    def write_texture_gpu(self, img):

        mapping = self.bufobj.map()
        dr.sync_device()
        cuda_driver.memcpy_dtod(mapping.device_ptr(),
                                img.data_ptr(), img.numel() * 4)
        mapping.unmap()

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, int(self.pbo))
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width,
                        self.height, GL_RGB, GL_FLOAT, ctypes.c_void_p(0))

    def end_frame(self):
        imgui.end()

        imgui.render()
        imgui.end_frame()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        self.impl.render(imgui.get_draw_data())
        self.impl.process_inputs()
        self.process_input()
        glfw.swap_buffers(self.window)
        glfw.poll_events()


if __name__ == "__main__":

    width, height = 1280, 720

    img = np.zeros((height, width, 3), dtype=np.float32)
    x = np.arange(0, 1, 1 / width)
    y = np.arange(0, 1, 1 / height)
    xx, yy = np.meshgrid(x, y)
    img[:, :, 0] = xx
    img[:, :, 1] = yy
    img[:, :, 2] = 0.5

    # img = torch.from_numpy(img).cuda()

    ui = UI(width, height, None)

    while not ui.should_close():
        ui.begin_frame()
        ui.write_texture_cpu(img)
        # ui.write_texture_gpu(img)
        ui.end_frame()

    ui.close()
