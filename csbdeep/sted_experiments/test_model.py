from __future__ import print_function, unicode_literals, absolute_import, division
from argparse import ArgumentParser
from os.path import join, exists, basename
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt

from csbdeep.utils import plot_some
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import CARE
from skimage.io import ImageCollection, imread, imsave
from skimage import img_as_float, img_as_ubyte

args = ArgumentParser()
args.add_argument("--base_dir")
args.add_argument("--model")
args.add_argument("--out_dir")
args.add_argument("--data")
#args.add_argument("base_dir")
#args.add_argument("model")
#args.add_argument("out_dir")
#args.add_argument("data")
args.add_argument("--is_3d", default=False)
args = args.parse_args()

model = CARE(config=None, name=args.model, basedir=args.base_dir)
#data = ImageCollection("training_data/val/low_snr_extracted_z/*.tif")
data = ImageCollection(args.data)
axes = "ZYX" if bool(args.is_3d) else "YX"

if not exists(args.out_dir):
    makedirs(args.out_dir)

for i in range(len(data)):
    im = img_as_float(data[i])
    r = model.predict(im, axes)
    #r = (r - r.min()) / (r.max() - r.min())
    r = img_as_ubyte(r)
    imsave(join(args.out_dir, f"{args.model}_{basename(data.files[i])}"), r)

