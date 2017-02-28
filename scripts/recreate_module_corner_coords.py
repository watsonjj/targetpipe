import numpy as np
from os.path import realpath, join, dirname
from targetpipe.io.pixels import pixel_sizes


x_pix_pos, y_pix_pos = np.load(join(realpath(dirname(__file__)),
                                    "../targetpipe/io/checm_pixel_pos.npy"))
pix_sizes = pixel_sizes(x_pix_pos, y_pix_pos)/2

x_pix_left = x_pix_pos - pix_sizes
x_pix_right = x_pix_pos + pix_sizes
y_pix_top = y_pix_pos + pix_sizes
y_pix_bottom = y_pix_pos - pix_sizes

module_l = np.min(x_pix_left.reshape((32, 64)), axis=1)
module_r = np.max(x_pix_right.reshape((32, 64)), axis=1)
module_t = np.max(y_pix_top.reshape((32, 64)), axis=1)
module_b = np.min(y_pix_bottom.reshape((32, 64)), axis=1)

module_x = np.column_stack([module_l, module_r, module_r, module_l])
module_y = np.column_stack([module_t, module_t, module_b, module_b])

module_tri1_x = np.column_stack([module_l, module_r, module_l])
module_tri2_x = np.column_stack([module_r, module_r, module_l])
module_tri1_y = np.column_stack([module_t, module_t, module_b])
module_tri2_y = np.column_stack([module_t, module_b, module_b])

module_tri_x = np.column_stack([module_tri1_x, module_tri2_x]).reshape((64, 3))
module_tri_y = np.column_stack([module_tri1_y, module_tri2_y]).reshape((64, 3))

module_square = np.stack([module_x, module_y])
module_tri = np.stack([module_tri_x, module_tri_y])

dir_ = realpath(dirname(__file__))
square_path = join(dir_, "../targetpipe/io/module_corner_coords.npy")
triangle_path = join(dir_, "../targetpipe/io/module_corner_tri_coords.npy")
np.save(square_path, module_square)
np.save(triangle_path, module_tri)
