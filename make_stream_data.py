#encoding:utf-8
import os
import csv
import json
import sys
import glob
import pandas as pd

if __name__ == '__main__':
  argv = sys.argv
  argc = len(argv)
  if (argc != 3):
    print('Usage: python %s import_file output_file' %argv[0])
    quit()
  input_file = open(argv[1], 'r', newline='')
  output_file = open(argv[2], 'w', newline='')
  csvWriter = csv.writer(output_file)
  
  input_data = csv.reader(input_file, doublequote=False)
  i = 0
  output_row = []
  no_point = [0.0]*50
  for row in input_data:
    row = [float(s) for s in row]
    category = int(row.pop(-1))
    if i == 0:
      before_row = row
    
    if row == no_point:
      output_row.extend(before_row)
      print("before_row:")
    else:
      output_row.extend(row)
      before_row = row

    if i == 9 :
      output_row.append(category)
      csvWriter.writerow(output_row)
      i = 0
      output_row = []
    else:
      i += 1

  input_file.close()
  output_file.close()    
