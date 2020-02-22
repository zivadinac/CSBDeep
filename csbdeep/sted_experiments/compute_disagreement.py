from __future__ import print_function, unicode_literals, absolute_import, division
from argparse import ArgumentParser
from os.path import join, exists, basename
from os import makedirs, walk
import numpy as np
import matplotlib.pyplot as plt

from csbdeep.utils import plot_some
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import CARE
from ensemble_disagreement import get_ensemble_disagreement
from skimage.io import ImageCollection, imread, imsave
from skimage.exposure import rescale_intensity
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import dilation
import tensorflow as tf

args = ArgumentParser()
args.add_argument("data_dir")
args.add_argument("--models_dir", default="care_probabilistic_models")
args.add_argument("--models", nargs='+', default="model_0 model_1 model_2 model_3 model_4")
args.add_argument("--out_dir", default="results")
args = args.parse_args()

for root, dirs, fs in walk(args.data_dir):
    data = [join(args.data_dir, f) for f in fs]

im_0 = imread(data[0])
ndim = len(im_0.shape)
axes = "ZYX" if ndim == 3 else "YX"
dil_kernel = np.ones((1,5,5,)) if ndim == 3 else np.ones((5,5))

models = [CARE(config=None, name=m, basedir=args.models_dir) for m in args.models.split()]

# disagreement on CPU due to GPU RAM limitations
# (several models are already loaded on the GPU)
#ed_sess = tf.Session(config=tf.ConfigProto(device_count = {"GPU":0}))
ed = get_ensemble_disagreement(ndim, len(models), im_0.shape, integration_method="trapezoidal")#, sess=ed_sess)
disagreement_perc = 90

makedirs(join(args.out_dir, "reconstruction"), exist_ok=True)
makedirs(join(args.out_dir, "disagreement"), exist_ok=True)
makedirs(join(args.out_dir, f"disagreement_{disagreement_perc}"), exist_ok=True)
makedirs(join(args.out_dir, "L_entropy"), exist_ok=True)
makedirs(join(args.out_dir, "l_entropy"), exist_ok=True)

for i in range(len(data)):
    im = imread(data[i])
    #im = img_as_float(data[i])
    out_fn = basename(data[i])

    predictions = [m.predict_probabilistic(im, axes) for m in models]
    
    r = np.mean(np.stack([p.mean() for p in predictions], ndim), ndim)
    r = (r - r.min()) / (r.max() - r.min())
    imsave(join(args.out_dir, "reconstruction", out_fn), img_as_ubyte(r))

    preds_mean_scale = [np.stack([p.mean(), p.scale()], ndim) for p in predictions]
    D, L_e, l_e = ed.eval(preds_mean_scale, return_entropies=True, mixture_k=20)

    print("D: ", D.min(), D.max())
    D = (D - D.min()) / (D.max() - D.min())
    imsave(join(args.out_dir, f"disagreement", out_fn), img_as_ubyte(D))

    D_perc = (D > np.percentile(D, disagreement_perc)).astype(np.float32)
    D_perc = dilation(D, dil_kernel)
    D_perc *= r
    imsave(join(args.out_dir, f"disagreement_{disagreement_perc}", out_fn), img_as_ubyte(D_perc))

    print("L_e: ", L_e.min(), L_e.max())
    imsave(join(args.out_dir, "L_entropy", out_fn), L_e)
    print("l_e: ", l_e.min(), l_e.max())
    imsave(join(args.out_dir, "l_entropy", out_fn), l_e)

