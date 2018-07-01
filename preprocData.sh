#!/bin/sh
in_dir="/data/wuyifan/MURA-v1.0/"
out_dir="/data/wuyifan/muraproc/"

python3 preprocess.py $in_dir $out_dir train.csv
cp $in_dir"train.csv" $out_dir"train.csv"

python3 preprocess.py $in_dir $out_dir valid.csv
cp $in_dir"valid.csv" $out_dir"valid.csv"
