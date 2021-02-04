import cv2
import numpy as np

img = cv2.imread("/home/samuel-moses/Pictures/graph.png", cv2.IMREAD_COLOR)
width, height, _ = img.shape

bboxes = np.random.randint(0, width, (7, 4), dtype='uint8')
p_tl = bboxes[:, 0, :]
p_br = bboxes[:, 3, :]

factor = 1 / 4

mod_dim = (int(width * factor), int(height * factor), 1)
p_tl, p_br = p_tl.astype(float), p_br.astype(float)
p_tl, p_br = p_tl * factor, p_br * factor
tl = p_tl.astype('uint8')
br = p_br.astype('uint8')

tl_heatmap, br_heatmap = np.zeros(mod_dim), np.zeros(mod_dim)

tl_reg = p_tl - tl
br_reg = p_br - br

tl_inds = tl[:, 1] * mod_dim[0] + tl[:, 0]
br_inds = br[:, 1] * mod_dim[0] + br[:, 0]

n, _ = tl.shape
