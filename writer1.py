from mayavi import mlab
import numpy as np
from tvtk.tools import visual
from drawnow import drawnow

from math import sqrt

def generate_random_data():
    data_shape = (8, 9, 8)
    x = np.zeros(data_shape)
    y = np.zeros(data_shape)
    z = np.zeros(data_shape)
    for i in range(0, 8):
        for j in range(0, 9):
            for k in range(0, 8):
                x[i, j, k] = 0.5 + i
                y[i, j, k] = 0.5 + j
                z[i, j, k] = 0.5 + k
    u = np.random.rand(data_shape[0], data_shape[1], data_shape[2]) - 0.5
    v = np.random.rand(data_shape[0], data_shape[1], data_shape[2]) - 0.5
    w = np.random.rand(data_shape[0], data_shape[1], data_shape[2]) - 0.5
    return x, y, z, u, v, w


class Writer:
    def __init__(self):
        self.data = []

    def visualize(self, data):
        self.data = data
        drawnow(self.draw)

    def draw(self):
        [x, y, z, u, v, w] = self.data
        fig=mlab.figure( bgcolor=(1, 1, 1) )
        visual.set_viewer(fig)
        for i in range(0, 8):
            for j in range(0, 9):
                for k in range(0, 8):
                    circle = self.__draw_circle(x[i, j, k], y[i, j, k], z[i, j, k], u[i, j, k], v[i, j, k], w[i, j, k])
        vectors = mlab.quiver3d(x, y, z, u, v, w, line_width = 3, scale_factor = 1, color = (0, 0, 1))
        mlab.show()

    def __draw_circle(self, x, y, z, u, v, w, r = 0.1, theta1 = 0, theta2 = np.pi * 2, start_color = (1, 0, 0), end_color = (0, 1, 0)):
		####====   Normalize Vector   ====####
        normal_x, normal_y, normal_z = self.__get_normal_vector(x, y, z, u, v, w)
		####====   Calculate Direction Vector 1   ====####
        vect1_x, vect1_y, vect1_z = self.__get_direction_vector1(normal_x, normal_y, normal_z)
		####====   Calculate Direction Vector 2   ====####
        vect2_x, vect2_y, vect2_z = self.__get_vector_product((normal_x, normal_y, normal_z), (vect1_x, vect1_y, vect1_z))
		####====   Calculate Circular Points   ====####
        rx, ry, rz = self.__get_circle_points((x, y, z), r, (vect1_x, vect1_y, vect1_z), (vect2_x, vect2_y, vect2_z), theta1 = theta1, theta2 = theta2, segments = 30)
        obj = mlab.plot3d(rx, ry, rz)
        return obj
		
    def __get_normal_vector(self, x, y, z, u, v, w):
        unit_x, unit_y, unit_z = u, v, w  
        length = sqrt(unit_x ** 2 + unit_y ** 2 + unit_z ** 2)
        return unit_x / length, unit_y / length, unit_z / length

    def __get_direction_vector1(self, normal_x, normal_y, normal_z):
        if normal_x <= normal_y and normal_x <= normal_z:
        	s = 1 / (normal_y ** 2 + normal_z ** 2)
        	return 0., s * normal_z, -1 * s * normal_y
        elif normal_y <= normal_x and normal_y <= normal_z:
        	s = 1 / (normal_x ** 2 + normal_z ** 2)
        	return s * normal_z, 0, -1 * s * normal_x
        else:
        	s = 1 / (normal_x ** 2 + normal_y ** 2)
        	return s * normal_y, -1 * s * normal_x, 0

    def __get_vector_product(self, vector1, normal):
        vect2_x = vector1[1] * normal[2] - vector1[2] * normal[1]
        vect2_y = vector1[2] * normal[0] - vector1[0] * normal[2]
        vect2_z = vector1[0] * normal[1] - vector1[1] * normal[0]
        return vect2_x, vect2_y, vect2_z
			
    def __get_circle_points(self, center, radius, vector1, vector2, theta1 = 0, theta2 = 2 * np.pi, segments = 300):
        theta = np.linspace(theta1, theta2, segments)
        rx = center[0] + radius * (vector1[0] * np.cos(theta) + vector2[0] * np.sin(theta))
        ry = center[1] + radius * (vector1[1] * np.cos(theta) + vector2[1] * np.sin(theta))
        rz = center[2] + radius * (vector1[2] * np.cos(theta) + vector2[2] * np.sin(theta))
        return rx, ry, rz

if __name__ == "__main__":
    x, y, z, u, v, w = generate_random_data()
    writer = Writer()
    writer.visualize([x, y, z, u, v, w])
    
