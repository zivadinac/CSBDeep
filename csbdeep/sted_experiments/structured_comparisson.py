from os import walk, makedirs
from os.path import join, basename
from argparse import ArgumentParser
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.applications import VGG16, VGG19

def grey_to_rgb_imitation(img):
    """ Append an axis to the 'img' and repeat three times along it.
        We need this in order to process grescaly image with RGB trained NN."""
    return np.repeat(img[...,np.newaxis], 3, -1)

args = ArgumentParser()
args.add_argument("--care_data")
args.add_argument("--gt_data")
args.add_argument("--out_dir")
args.add_argument("--net", default="VGG16")
args.add_argument("--img_size", default="320x320")
args.add_argument("--layer", default="block5_conv3")
args = args.parse_args()

args.care_data = "training_data/first_set/care"
args.gt_data = "training_data/first_set/gt"
args.layer = "block4_conv3"
args.out_dir = f"training_data/first_set/feature_diffs_{args.layer}"

makedirs(args.out_dir, exist_ok=True)

img_size = args.img_size.split('x')
img_size = (int(img_size[0]), int(img_size[1]))

vggNetworkType = VGG16 if args.net == "VGG16" else VGG19
img_in = Input(shape=(*img_size, 3), dtype="float32", name="img_in")
network = vggNetworkType(include_top=False, input_tensor=img_in, pooling=None)
model = Model(inputs=img_in, outputs=network.get_layer(args.layer).output)

#for l in network.layers:
#    print(l.name, l.output_shape)

for root, dirs, fs in walk(args.gt_data):
    gt_data = [join(args.gt_data, f) for f in fs]
    gt_data.sort()

for root, dirs, fs in walk(args.care_data):
    care_data = [join(args.care_data, f) for f in fs]
    care_data.sort()

assert len(gt_data) == len(care_data)

for i in range(len(gt_data)):
    #print(basename(care_data[i]), "   ", basename(gt_data[i]))
    #continue
    care_img = imread(care_data[i])
    gt_img = imread(gt_data[i])
    assert care_img.shape == gt_img.shape

    care_features = model.predict(grey_to_rgb_imitation(care_img))
    gt_features = model.predict(grey_to_rgb_imitation(gt_img))

    feature_diff = np.linalg.norm(care_features - gt_features, axis=-1)
    feature_diff_rs = resize(feature_diff, (feature_diff.shape[0], *img_size))
    imsave(join(args.out_dir, f"image_{i}.tif"), feature_diff_rs)
    plt.hist(feature_diff.flatten())
    plt.savefig(f"fd_hist_{args.layer}.png")
    print(f"Finished {i}th image.")

