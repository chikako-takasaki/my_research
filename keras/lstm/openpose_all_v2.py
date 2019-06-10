import time, sys
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#argv = sys.argv 
#layer_num = int(argv[1])
#nb_hidden = int(argv[2])
#dropout = float(argv[1])
#bn = argv[2]
#image_name = argv[3]

def preprocessing(data):
  column_len = len(data.columns)
  for i in range(column_len):
    values = data[i].values.reshape(-1,1)
    ss = StandardScaler()
    ss.fit(values)
    v_std = ss.transform(values)
    mms = MinMaxScaler()
    v_norm = mms.fit_transform(values)
    data.loc[:,i] = v_norm
  return data.values

# パラメーター
length_of_sequence = 10 
in_out_neurons = 50
n_hidden = 100

# データ整形
data = pd.read_csv('../all_stream.csv', header=None)
column_len = len(data.columns)
X = data.loc[:, 0 : (column_len-2)]
y = data.loc[:, column_len-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("*********************************************")
#X_train = X_train.values
#X_train /= 1920.0
#X_test = X_test.values
#X_test /= 1920.0
X_train = preprocessing(X_train)
X_test = preprocessing(X_test)
y_train = y_train.values
y_test = y_test.values

num_classes = 3
X_train = numpy.reshape(X_train, (X_train.shape[0], length_of_sequence, in_out_neurons))
X_test = numpy.reshape(X_test, (X_test.shape[0], length_of_sequence, in_out_neurons))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model(activation="relu"):
  model = Sequential([
    TimeDistributed(Dense(n_hidden), input_shape=(length_of_sequence, in_out_neurons)),
    TimeDistributed(Dense(n_hidden)),
    LSTM(n_hidden, return_sequences=True, dropout=0.2, input_shape=(length_of_sequence, in_out_neurons)),
    LSTM(n_hidden),
    Dense(num_classes),
    Activation('sigmoid')
  ])
   
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
fig.savefig("./image/allv2_do02_100.png")
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



