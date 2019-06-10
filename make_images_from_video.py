#coding:utf-8
import os
import sys
import glob
import subprocess
import ffmpeg
from numpy.random import *

def get_images(file_name, output_dir, num):
  for i in range(2):
    output_file = output_dir + str(num+1) + str(i+1) + "%02d.jpg"
    
    stream = ffmpeg.input(file_name)
    stream = ffmpeg.output(stream, output_file, t=2, ss=1, r=20 ,f='image2')   
    print(file_name)
    print(output_file)
    print(stream)
    ffmpeg.run(stream)

if __name__ == '__main__':
  argv = sys.argv
  argc = len(argv)
  if (argc != 3):
    print('Usage: python %s import_dir output_dir' %argv[0])
    quit()
  file_names = glob.glob(argv[1] + '*.mp4')
  output_dir = argv[2]
  num = 0
  for file_name in file_names:
    get_images(file_name, output_dir, num)
    num += 1 
