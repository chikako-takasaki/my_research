import numpy as np
import pandas as pd

def get_rotation_matrix(rad):
  rot = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad), np.cos(rad)]])
  return rot

def rotation_augment(data, deg):
  rad = deg * np.pi / 180
  rot = get_rotation_matrix(rad)
  new_data = data.copy()
  for i in range(len(new_data)):
    row = new_data.iloc[i]
    for c in range(len(new_data.columns)):
      if c % 2 == 0 and c != len(new_data.columns)-1:
        col = np.array([row.iloc[c], row.iloc[c+1]])
        new_col = np.dot(rot, col)
        new_data.iloc[i,c] = new_col[0]
        new_data.iloc[i,c+1] = new_col[1]
  return new_data


def reverse_augment(data):
  new_data = data.copy()
  rev = np.array([[-1, 0],[0, 1]])
  for i in range(len(new_data)):
    row = new_data.iloc[i]
    for c in range(len(new_data.columns)):
      if c % 2 == 0 and c != len(new_data.columns)-1:
        col = np.array([row.iloc[c], row.iloc[c+1]])
        new_col = np.dot(rev, col)
        new_data.iloc[i,c] = new_col[0]
        new_data.iloc[i,c+1] = new_col[1]
  return new_data
 
