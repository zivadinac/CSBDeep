import numpy as np

from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from ensemble_disagreement import tiled_disagreement

dis = imread("results_3d_prob_0_255/disagreement/Reslice_of_20200116_SUSHI_CARE_150microM_TRAINING_08.tif")
im = imread("training_data/2nd_set/val/low_snr/Reslice_of_20200116_SUSHI_CARE_150microM_TRAINING_08.tif")
perc_ts = [70, 80, 90, 95]
occ_ts = [0.2, 0.3, 0.4, 0.5, 0.6]

for p in perc_ts:
    for o in occ_ts:
        dim = tiled_disagreement(dis, (1, 50, 50), p, o)
        print(f"{p}_{o}: ", np.sum(dim) / np.prod(dim.shape))
        dim = dim.astype(np.int8) * im
        imsave(f"tiled_disagreements/perc_{p}_occ_{o}.tif", dim)

#dim = tiled_disagreement(im, (1, 50, 50), 60, 0.5)
#dim = dim.astype(np.int8) * 255
