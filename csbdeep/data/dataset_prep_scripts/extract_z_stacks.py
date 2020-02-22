from skimage.io import imread, imsave
from os import walk, mkdir
import os.path as path
import argparse

args = argparse.ArgumentParser()
args.add_argument("input_dir")
args.add_argument("output_dir")
args = args.parse_args()

if not path.exists(args.output_dir):
    mkdir(args.output_dir)

for root, dirs, files in walk(args.input_dir):
    for f in files:
        im = imread(path.join(args.input_dir, f))
        for i in range(im.shape[0]):
            r, ext = path.splitext(f)
            im_i_out_path = path.join(args.output_dir, r + "_" + str(i) + ext)
            imsave(im_i_out_path, im[i])
