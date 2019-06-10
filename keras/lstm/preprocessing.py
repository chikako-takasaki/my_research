import pandas as pd
import numpy as np

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

  data[column_len-1] = label.values
  return data
