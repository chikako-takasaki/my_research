#!/bin/bash

mkdir -p $2
for file in `\find $1 -maxdepth 1 -name '*.jpg'`; do
  file_name=`basename $file`
  out_file="$2${file_name}"
  echo $out_file
  ./ffmpeg -i ${file} -vf "rotate=2*PI/180" ${out_file}
done
exit 0
