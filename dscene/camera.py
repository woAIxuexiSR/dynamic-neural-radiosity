import numpy as np


class FPSCamera:

    # np.ndarray (4, 4)
    def __init__(self, transform, speed):
        self.pos = transform[:3, 3]
        self.x = transform[:3, 0]
        self.y = transform[:3, 1]
        self.z = transform[:3, 2]

        self.phi = np.rad2deg(np.arctan2(self.z[2], self.z[0]))
        self.theta = np.rad2deg(np.arccos(self.z[1]))
        self.speed = speed

    def get_transform(self):
        return np.array([
            [self.x[0], self.y[0], self.z[0], self.pos[0]],
            [self.x[1], self.y[1], self.z[1], self.pos[1]],
            [self.x[2], self.y[2], self.z[2], self.pos[2]],
            [0, 0, 0, 1]
        ])

    # idx: 0 for z, 1 for y, 2 for x
    def move(self, idx, delta):
        if idx == 0:
            self.pos += delta * self.z * self.speed
        elif idx == 1:
            self.pos += delta * self.y * self.speed
        else:
            self.pos += delta * self.x * self.speed

    def rotate(self, x, y):
        self.theta = np.clip(self.theta + y, 1, 179)
        self.phi += x

        cos_theta = np.cos(np.deg2rad(self.theta))
        sin_theta = np.sin(np.deg2rad(self.theta))
        cos_phi = np.cos(np.deg2rad(self.phi))
        sin_phi = np.sin(np.deg2rad(self.phi))

        self.z = np.array(
            [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi])
        self.x = np.array([sin_phi, 0, -cos_phi])
        self.y = np.array(
            [-cos_theta * cos_phi, sin_theta, -cos_theta * sin_phi])


class MovingCamera:

    # np.ndarray (3,) (3,) (4,) (4,)    quaternion in xyzw
    def __init__(self, p1, p2, r1, r2):
        self.pos_start = np.array(p1)
        self.pos_end = np.array(p2)
        self.rot_start = np.array(r1) / np.linalg.norm(np.array(r1))
        self.rot_end = np.array(r2) / np.linalg.norm(np.array(r2))
        
    def qslerp(self, q1, q2, t):
        
        cos_theta = np.dot(q1, q2)
        
        if cos_theta < 0:
            q1 = -q1
            cos_theta = -cos_theta
            
        angle = np.arccos(cos_theta)
        return (np.sin((1 - t) * angle) * q1 + np.sin(t * angle) * q2) / np.sin(angle)

    def get_transform(self, t):

        pos = self.pos_start + (self.pos_end - self.pos_start) * t
        rot = self.qslerp(self.rot_start, self.rot_end, t)

        return np.array([
            [1 - 2 * rot[1]**2 - 2 * rot[2]**2, 2 * rot[0] * rot[1] - 2 * rot[2]
                * rot[3], 2 * rot[0] * rot[2] + 2 * rot[1] * rot[3], pos[0]],
            [2 * rot[0] * rot[1] + 2 * rot[2] * rot[3], 1 - 2 * rot[0]**2 -
                2 * rot[2]**2, 2 * rot[1] * rot[2] - 2 * rot[0] * rot[3], pos[1]],
            [2 * rot[0] * rot[2] - 2 * rot[1] * rot[3], 2 * rot[1] * rot[2] +
                2 * rot[0] * rot[3], 1 - 2 * rot[0]**2 - 2 * rot[1]**2, pos[2]],
            [0, 0, 0, 1]
        ])
