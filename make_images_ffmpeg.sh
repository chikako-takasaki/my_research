#!/bin/bash

n=1
for file in `\find $1 -maxdepth 1 -name '*.mp4'`; do
  out_file=`printf $2%04d $n`
  out_file="${out_file}%02d.jpg"
  echo ${out_file}
  ./ffmpeg -i ${file} -ss 0 -t 3 -f image2 -vf fps=10 ${out_file} 
  n=$((n + 1))
done
exit 0
