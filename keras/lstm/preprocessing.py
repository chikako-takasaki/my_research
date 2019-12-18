import pandas as pd
import numpy as np
import sys
import csv

def preprocessing(input_data):
  column_len = len(input_data.columns)
  data = input_data.loc[:, 0 : (column_len-2)]
  label = input_data.loc[:, column_len-1]
  data_len = len(data)
  for i in range(data_len):
    data_row = data.iloc[i]
    max_v = max(data_row)
    min_v = min(data_row)
    for c in range(data_row.shape[0]):
      v = data_row.values[c]
      new_v = (v - min_v) / (max_v - min_v)
      data.iloc[i,c] = new_v
    if(i%10000 == 0):
      print(i)

  data[column_len-1] = label.values
  return data

if __name__ == '__main__':
    argv = sys.argv
    read_csv = argv[1]
    write_csv = argv[2]
    data = pd.read_csv(read_csv, header=None)
    pre = preprocessing(data)
    pre.to_csv(write_csv, header=False, index=False)
