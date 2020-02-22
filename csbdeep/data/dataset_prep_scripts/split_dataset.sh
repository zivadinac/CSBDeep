#!/bin/bash

in_dir="$1"
out_dir="$2"
val_images="${@:3:99}"

echo "$in_dir"
echo "$out_dir"
echo "$val_images"

mkdir "$out_dir/train"
mkdir "$out_dir/train/low_snr"
mkdir "$out_dir/train/high_snr"

mkdir "$out_dir/val"
mkdir "$out_dir/val/low_snr"
mkdir "$out_dir/val/high_snr"

mv "$in_dir/$val_images" "$out_dir/val"
mv "$out_dir/*_HIGH.tif" "$out_dir/val/high_snr"
mv "$out_dir/*_LOW.tif" "$out_dir/val/low_snr"

mv "$in_dir/*_HIGH.tif" "$out_dir/train/high_snr"
mv "$in_dir/*_LOW.tif" "$out_dir/train/low_snr"
