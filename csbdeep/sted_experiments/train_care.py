from __future__ import print_function, unicode_literals, absolute_import, division
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread, imsave, ImageCollection
from skimage import img_as_float

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.data import RawData

args = ArgumentParser()
args.add_argument("train_data")
args.add_argument("val_data")
args.add_argument("--num_models", default=5)
args.add_argument("--epochs", default=100)
args.add_argument("--steps_per_epoch", default=400)
args.add_argument("--batch_size", default=32)
args.add_argument("--probabilistic", default=1)
args.add_argument("--is_3d", default=False)
args.add_argument("--models_dir", default="care_probabilistic_models")
args = args.parse_args()

tr_data, _, tr_axes = load_training_data(args.train_data, validation_split=0)
val_data, _, val_axes = load_training_data(args.val_data, validation_split=0)
#val_data = np.load(args.val_data)
#val_data = (val_data["X"], val_data["Y"])
#val_axes = tr_axes # we assume that both training and validation data are saved in the same format

is_3d = bool(args.is_3d)
axes = "ZYX" if is_3d else "YX"
n_dim = 3 if is_3d else 2
config = Config(axes, n_dim=n_dim, n_channel_in=1, n_channel_out=1, probabilistic=int(args.probabilistic), train_batch_size=int(args.batch_size), unet_kern_size=3) 

for i in range(int(args.num_models)):
    model = CARE(config, f"model_{i}", args.models_dir)
    train_history = model.train(tr_data[0], tr_data[1], validation_data=val_data,
                                epochs=int(args.epochs),
                                steps_per_epoch=int(args.steps_per_epoch))

exit()

plot_history(train_history, ["loss", "val_loss"], ["mse", "val_mse", "mae", "val_mae"])

def crop_to_even(img):
    shape = img.shape
    new_shape = (shape[0] - (shape[0] % 2), shape[1] - (shape[1] % 2))
    return img[0:new_shape[0], 0:new_shape[1]]

val_data_full_ims = ImageCollection("training_data/val/low_snr_extracted_z/*.tif")
for i in range(len(val_data_full_ims)):
    x = crop_to_even(img_as_float(val_data_full_ims[i]))
    x = np.reshape(x, (1, *x.shape, 1))
    y_ = model.keras_model.predict(x)[0,:,:,1]
    imshow(y_)
    plt.show()

