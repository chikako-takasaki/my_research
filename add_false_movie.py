import glob
import sys
import pandas as pd

w_removed = [23, 63, 80, 89, 231, 286, 568]
r_removed = [562, 785]
b_removed = [38, 86, 197, 260, 359, 394, 438, 453, 724, 846, 955, 990, 1108]

def add_movie_num(input_file, out_file):
  data = pd.read_csv(input_file)
  w_files = sorted(glob.glob('/mnt/nas/STAIR_Actions_v1.0/writing/*'))
  r_files = sorted(glob.glob('/mnt/nas/STAIR_Actions_v1.0/reading_newspaper/*'))
  b_files = sorted(glob.glob('/mnt/nas/STAIR_Actions_v1.0/bowing/*'))
  
  w_num = len(w_files) - len(w_removed)
  r_num = len(r_files) - len(r_removed)
  b_num = len(b_files) - len(b_removed)
  movie_num = []
  for i in w_removed:
    w_files.pop(i-1)

  for i in r_removed:
    r_files.pop(i-1)

  for i in b_removed:
    b_files.pop(i-1)

  length = len(data)
  for i in range(length):
    print(i)
    row = data.iloc[i, :]
    if row[2] == 0:
     num = w_files[row[0]-1]
    elif row[2] == 1:
     num = r_files[row[0]-w_num-1]
    else:
     num = b_files[row[0]-w_num-r_num-1]
    movie_num.append(num)
  data['movie_num'] = movie_num
  data.to_csv(out_file, index=False)


if __name__ == '__main__':
  argv = sys.argv
  add_movie_num(argv[1], argv[2])
