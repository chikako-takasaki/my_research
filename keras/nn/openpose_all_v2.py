import time
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import  BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

import pandas as pd
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, GroupKFold
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import numpy

#argv = sys.argv 
#layer_num = int(argv[1])
#nb_hidden = int(argv[2])
#dropout = float(argv[1])
#bn = argv[2]
#image_name = argv[3]

data = pd.read_csv('../output_all_v2.csv', header=None)
column_len = len(data.columns)
X_w = data[data.loc[:, column_len-1]==0]
X_r = data[data.loc[:, column_len-1]==1]
X_b = data[data.loc[:, column_len-1]==2]

w_num = len(X_w)
r_num = len(X_r)
b_num = len(X_b)

w_train = X_w[0:4530] 
w_test = X_w[4530:w_num]
r_train = X_r[0:6190] 
r_test = X_r[6190:r_num]
b_train = X_b[0:7860] 
b_test = X_b[7860:b_num]

all_train = w_train.append(r_train.append(b_train))
all_test = w_test.append(r_test.append(b_test))
X_train = all_train.loc[:, 0 : 49]
X_test = all_test.loc[:, 0 : 49]
y_train = all_train.loc[:, 50]
y_test = all_test.loc[:, 50]

X_train = X_train.values
X_train /= 1920.0
X_test = X_test.values
X_test /= 1920.0
y_train = y_train.values
y_test = y_test.values

group_index = []
for i in range(1858):
  group_index = group_index + [i]*10

num_classes = 3
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model(activation="relu", nb_hidden=50, layer_num=3, dropout=0.2, bn='false'):
  model = Sequential([
    Dense(50, input_shape=(50,)),
    Activation(activation),
#    Dropout(dropout),
  ])
  for i in range(layer_num) :
    model.add(Dense(nb_hidden))
#    model.add(BatchNormalization())
    model.add(Activation(activation))
#    model.add(Dropout(dropout))
   
  model.add(Dense(3))
  model.add(Activation('sigmoid'))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  return model

def plot_history_loss(fit6):
  # Plot the loss in the history
  axL.plot(fit6.history['loss'],label="loss for training")
  axL.plot(fit6.history['val_loss'],label="loss for validation")
  axL.set_title('model loss')
  axL.set_xlabel('epoch')
  axL.set_ylabel('loss')
  axL.set_ylim([0.0,1.6])
  axL.legend(loc='upper right')

def plot_history_acc(fit6):
  # Plot the loss in the history
  axR.plot(fit6.history['acc'],label="accuracy for training")
  axR.plot(fit6.history['val_acc'],label="accuracy for validation")
  axR.set_title('model accuracy')
  axR.set_xlabel('epoch')
  axR.set_ylabel('accuracy')
  axR.set_ylim([0.4,1.0])
  axR.legend(loc='lower right')

model = create_model()
model.summary()
  
# Fit the model
start = time.time()
fit = model.fit(X_train, y_train, verbose=2, epochs=1600, validation_data=(X_test, y_test))
process_time = time.time() - start

# Evaluate
print("model training time")
print(process_time)

train_score = model.evaluate(X_train, y_train, verbose=0)
print("train loss:",train_score[0])
print('train accuracy : ', train_score[1])

test_score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_score[0])
print('test accuracy : ', test_score[1])

# plot 
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig("allv2.png")
plt.close()



#kfold = GroupKFold(n_splits=1)
#cvscores = []
"""
n=1
for train, test in kfold.split(X_train, y_train, group_index):
  # Create model
  model = create_model(dropout=dropout, bn=bn)
  model.summary()
  
  # Fit the model
  start = time.time()
  fit = model.fit(X_train[train], y_train[train], verbose=2, epochs=1600, validation_data=(X_test, y_test))
  process_time = time.time() - start

  # Evaluate
  scores = model.evaluate(X_train[test], y_train[test], verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
  cvscores.append(scores[1])

  print("model training time")
  print(process_time)

  train_score = model.evaluate(X_train[train], y_train[train], verbose=0)
  print("train loss:",train_score[0])
  print('train accuracy : ', train_score[1])

  test_score = model.evaluate(X_train[test], y_train[test], verbose=0)
  print("Test loss:", test_score[0])
  print('test accuracy : ', test_score[1])

  # plot 
  fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
  plot_history_loss(fit)
  plot_history_acc(fit)
  fig.savefig('./image/all5_75v2dobn_1600_%d.png' % n)
  plt.close()

  n+=1

print("%.3f (+/- %.3f)" % (numpy.mean(cvscores), numpy.std(cvscores)))
"""
#start = time.time()
#grid_result = grid.fit(X_train, y_train, verbose=2, epochs=800, validation_data=(X_test, y_test))
#process_time = time.time() - start
#print (grid_result.best_score_)
#print (grid_result.best_params_)



