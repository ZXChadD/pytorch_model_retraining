import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 600
image_mean = np.array([127, 127, 127]) # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(32, 16, SSDBoxSizes(20, 35), [2, 3]),
    SSDSpec(16, 32, SSDBoxSizes(35, 50), [2, 3]),
    SSDSpec(8, 64, SSDBoxSizes(50, 65), [2, 3]),
    SSDSpec(4, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 600), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)