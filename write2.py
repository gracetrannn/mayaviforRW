from mayavi import mlab
import numpy as np
from tvtk.tools import visual
from drawnow import drawnow

from math import sqrt

data_shape = (4, 5, 4)


def generate_random_data():
    x = np.zeros(data_shape)
    y = np.zeros(data_shape)
    z = np.zeros(data_shape)
    for i in range(0, 8, 2):
        for j in range(0, 9, 2):
            for k in range(0, 8, 2):
                ii, jj, kk = int(i / 2), int(j / 2), int(k / 2)
                x[ii, jj, kk], y[ii, jj, kk], z[ii, jj, kk] = 1 + i, 1 + j, 1 + k
    u = np.random.rand(data_shape[0], data_shape[1], data_shape[2]) - 1
    v = np.random.rand(data_shape[0], data_shape[1], data_shape[2]) - 1
    w = np.random.rand(data_shape[0], data_shape[1], data_shape[2]) - 1
    return x, y, z, u, v, w


def orthogonal_vector(vector):
    vector = np.array(vector)
    if vector[0] == 0 and vector[1] == 0:
        orthogonal = np.array([1, 0, 0])
    else:
        orthogonal = np.array([vector[1], -vector[0], 0])
    orthogonal_vector = np.cross(vector, orthogonal)
    unit_u, unit_v, unit_w = orthogonal_vector[0], orthogonal_vector[1], orthogonal_vector[2]
    length = sqrt(unit_u ** 2 + unit_v ** 2 + unit_w ** 2)
    if length == 0:
        return [0, 0, 0]
    else:
        return [unit_u / length, unit_v / length, unit_w / length]


class Writer:
    def __init__(self):
        self.data = []

    def visualize(self, data):
        self.data = data
        drawnow(self.draw)

    def draw(self):
        [x, y, z, u, v, w] = self.data
        fig = mlab.figure(bgcolor=(1, 1, 1))
        visual.set_viewer(fig)
        for i in range(0, 4):
            for j in range(0, 5):
                for k in range(0, 4):
                    vector1 = [u[i, j, k], v[i, j, k], w[i, j, k]]
                    orthogonal_vect = orthogonal_vector(vector1)
                    circle1 = self.__draw_circle(
                        x[i, j, k], y[i, j, k], z[i, j, k], vector1, orthogonal_vect, color=(1, 0, 0))
                    vector1 = mlab.quiver3d(x[i, j, k], y[i, j, k], z[i, j, k], u[i, j, k],
                                            v[i, j, k], w[i, j, k], line_width=3, scale_factor=1, color=(1, 0, 0))
                    vector2 = mlab.quiver3d(
                        x[i, j, k], y[i, j, k], z[i, j, k], orthogonal_vect[0], orthogonal_vect[1], orthogonal_vect[2], line_width=3, scale_factor=1, color=(0, 0, 1))
        mlab.show()

    def __draw_circle(self, x, y, z, vector1, vector2, r=0.03*10, theta1=0, theta2=np.pi * 2, color=(0, 0, 1)):
        #### ====   Calculate Normal Vector of the Plane   ====####
        normal_x, normal_y, normal_z = self.__get_vector_product(
            vector1, vector2)
        #### ====   Normalize Normal Vector   ====####
        normal_x, normal_y, normal_z = self.__get_normal_vector(
            normal_x, normal_y, normal_z)
        #### ====   Calculate Direction Vector 1 in the Plane ====####
        # Use the first vector directly as one direction vector in the plane
        vect1_x, vect1_y, vect1_z = vector1
        #### ====   Calculate Direction Vector 2 in the Plane ====####
        # Use the second vector directly as the other direction vector in the plane
        vect2_x, vect2_y, vect2_z = vector2

        #### ====   Calculate Circular Points   ====####
        rx, ry, rz = self.__get_circle_points((x, y, z), r, (vect1_x, vect1_y, vect1_z), (
            vect2_x, vect2_y, vect2_z), theta1=theta1, theta2=theta2, segments=30)
        obj = mlab.plot3d(rx, ry, rz, color=color,
                          tube_radius=0.01*3, line_width=0.01)
        return obj

    def __get_normal_vector(self, x, y, z):
        unit_x, unit_y, unit_z = x, y, z
        length = sqrt(unit_x ** 2 + unit_y ** 2 + unit_z ** 2)
        if length == 0:
            return 0, 0, 0
        else:
            return unit_x / length, unit_y / length, unit_z / length

    def __get_direction_vector1(self, normal_x, normal_y, normal_z):
        norm_xy = normal_x ** 2 + normal_y ** 2
        norm_xz = normal_x ** 2 + normal_z ** 2
        norm_yz = normal_y ** 2 + normal_z ** 2
        if norm_xy == 0 and norm_yz == 0 and norm_xz == 0:
            return 0, 0, 0
        elif normal_x <= normal_y and normal_x <= normal_z:
            s = 1 / norm_yz
            return 0., s * normal_z, -1 * s * normal_y
        elif normal_y <= normal_x and normal_y <= normal_z:
            s = 1 / norm_xz
            return s * normal_z, 0, -1 * s * normal_x
        else:
            s = 1 / norm_xy
            return s * normal_y, -1 * s * normal_x, 0

    def __get_vector_product(self, vector1, normal):
        vect2_x = vector1[1] * normal[2] - vector1[2] * normal[1]
        vect2_y = vector1[2] * normal[0] - vector1[0] * normal[2]
        vect2_z = vector1[0] * normal[1] - vector1[1] * normal[0]
        return vect2_x, vect2_y, vect2_z

    def __get_circle_points(self, center, radius, vector1, vector2, theta1=0, theta2=2 * np.pi, segments=300):
        theta = np.linspace(theta1, theta2, segments)
        rx = center[0] + radius * \
            (vector1[0] * np.cos(theta) + vector2[0] * np.sin(theta))
        ry = center[1] + radius * \
            (vector1[1] * np.cos(theta) + vector2[1] * np.sin(theta))
        rz = center[2] + radius * \
            (vector1[2] * np.cos(theta) + vector2[2] * np.sin(theta))
        return rx, ry, rz


if __name__ == "__main__":
    data = np.load('data1.npy')
    [x, y, z, u, v, w] = data
    writer = Writer()
    writer.visualize(data)
    # generate_random_data()
