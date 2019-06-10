import copy

def replace_zeros(data):
  new_data = []
  drop_flag = False
  for i in range(len(data)):
    v = data[i]
    if i % 10 == 0:
      drop_flag = False
      if v == 0:
        n = 1
        flag = False
        while flag == False:
          after_v = data[i+n]
          if after_v != 0:
            new_data.append(after_v)
            flag = True
          n+=1
          if n==10:
            print('data drop')
            flag = True
            drop_flag = True
      else:
        new_data.append(v)
    else:
      if v == 0 and drop_flag == False:
        new_data.append(new_data[-1])
      elif v != 0:
        new_data.append(v)
  print(new_data)
