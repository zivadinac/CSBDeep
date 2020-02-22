from argparse import ArgumentParser
from csbdeep.data import RawData, create_patches

args = ArgumentParser()
args.add_argument("input_basepath")
args.add_argument("input_x")
args.add_argument("input_y")
args.add_argument("output_file")
args.add_argument("--patch_size", default=64)
args.add_argument("--n_patches_per_image", default=100)
args.add_argument("--axes", default="YX")
args = args.parse_args()

data = RawData.from_folder(basepath=args.input_basepath,
                                    source_dirs=[args.input_x],
                                    target_dir=args.input_y,
                                    axes=args.axes)

ps = int(args.patch_size)
X, Y, axes = create_patches(data,
                            patch_size=(ps, ps, 1),
                            n_patches_per_image=int(args.n_patches_per_image),
                            save_file=args.output_file,
                            patch_axes=args.axes+"C")
