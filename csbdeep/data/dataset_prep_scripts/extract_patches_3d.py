from argparse import ArgumentParser
from csbdeep.data import RawData, create_patches

args = ArgumentParser()
args.add_argument("input_basepath")
args.add_argument("input_x")
args.add_argument("input_y")
args.add_argument("output_file")
args.add_argument("--patch_size_xy", default=32)
args.add_argument("--patch_size_z", default=16)
args.add_argument("--n_patches_per_image", default=750)
args.add_argument("--axes", default="ZYX")
args = args.parse_args()

data = RawData.from_folder(basepath=args.input_basepath,
                                    source_dirs=[args.input_x],
                                    target_dir=args.input_y,
                                    axes=args.axes)

ps_xy = int(args.patch_size_xy)
ps_z = int(args.patch_size_z)

X, Y, axes = create_patches(data,
                            patch_size=(ps_z, ps_xy, ps_xy, 1),
                            n_patches_per_image=int(args.n_patches_per_image),
                            save_file=args.output_file,
                            patch_axes=args.axes+"C")
