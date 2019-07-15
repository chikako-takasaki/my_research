import os
import csv
import json
import sys
import glob
import pandas as pd

category_dict = { 'writing': 0, 'reading': 1, 'bowing': 2 }

def json_to_series(n,input_file, category_name):
  json_data = json.load(input_file)
  category = category_dict[category_name]
  try:
    pose_key_points = json_data['people'][0]['pose_keypoints_2d']
    point_len = len(pose_key_points) // 3
    data_list = []
    for i in range(0, point_len):
      data_list.append(pose_key_points[i*3])
      data_list.append(pose_key_points[i*3+1])
    data_list.append(category)
  except(IndexError):
    print(n)
    data_list = [0] * 50
    data_list.append(category)
 
  return pd.Series(data_list)


def make_data_per_category(import_dir, category_name):
  n = 0
  input_filenames = sorted(glob.glob(import_dir + '*.json'))
  data = pd.DataFrame(columns=list(range(0, 51)))
  for input_filename in input_filenames:
    input_file = open(input_filename, 'r')
    row = json_to_series(n, input_file, category_name)
    data = data.append(row, ignore_index=True)
    input_file.close()
    n += 1
  return data  

def replace_zeros(data):
  data_len = len(data)
  column_len = len(data.columns)
  new_data = pd.DataFrame(columns=list(range(0, 51)))
  for i in range(data_len):
    row = data.iloc[i, :column_len-1]
    category = data.iloc[i, column_len-1]
    if i % 10 == 0:
      drop_flag = False
      if len(row[row==0]) == column_len-1:
        n = 1
        flag = False
        while flag == False:
          after_row = data.iloc[i+n, :column_len-1]
          if len(row[row==0]) != column_len-1:
            after_row[column_len-1] = category
            new_data = new_data.append(after_row)
            flag=True
          n += 1
          if n == 10:
            flag = True
            drop_flag = True
            print('category:{}, index:{} removed.'.format(category, i))
      else:
        row[column_len-1] = category
        new_data = new_data.append(row)
    else:
      if len(row[row==0]) == column_len-1 and drop_flag == False:
        new_data = new_data.append(new_data.iloc[-1])
       
      elif len(row[row==0]) != column_len-1:
        row[column_len-1] = category
        new_data = new_data.append(row)
  return new_data

def make_all_data(base_dir, output_filename):
  dir_list = glob.glob(base_dir+'*')
  all_data = pd.DataFrame(columns=list(range(0, 51)))
  for dire in dir_list:
    if os.path.isdir(dire):
      data_per_category = make_data_per_category(dire, os.path.basename(dire))
    all_data = all_data.append(data_per_category)
  all_data.to_csv(output_filename+'pre.csv', header=False, index=False) 
  
  new_all_data = replace_zeros(all_data)
  new_all_data.to_csv(output_filename+'.csv', header=False, index=False)

if __name__ == '__main__':
  argv = sys.argv
  argc = len(argv)
  if (argc != 4):
    print('Usage: python %s import_dir output_file category_name' %argv[0])
    quit()
  input_filenames = sorted(glob.glob(argv[1] + '*.json'))
  output_file = open(argv[2], 'a', newline='')
  category_name = argv[3]

  if not os.path.exists("./" + argv[2]):
    init_csv(output_file)
 
  n=0
  for input_filename in input_filenames:
    print(input_filename)
    input_file = open(input_filename, 'r')
    output_json(input_file, output_file, category_name)  
    input_file.close()
    n += 1

  print(n)
  output_file.close()
