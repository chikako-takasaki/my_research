import time, sys
from keras.models import Sequential
from  keras.layers import Reshape, Flatten, LSTM, TimeDistributed, Dense, Activation, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping 

import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import preprocessing
import augmentation
# パラメーター
length_of_sequence = 10 
in_out_neurons = 50
n_hidden = 50
"""
# データ整形
data = pd.read_csv('~/work/stream_s10.csv', header=None)
column_len = len(data.columns)
train, test = train_test_split(data, random_state=0)
"""
train = pd.read_csv('~/work/keras/lstm/train_aug_normalized6.csv', header=None) 
test = pd.read_csv('~/work/keras/lstm/test_aug_normalized6.csv', header=None) 
column_len = len(train.columns)
"""
# train augmentation
start = time.time()
train_rev = augmentation.reverse_augment(train)
process_time = time.time() - start
print('train reverse aug done')
print(process_time)
start = time.time()
train_rev5 = augmentation.rotation_augment(train_rev, 5)
process_time = time.time() - start
print('train reverse rotate5 aug done')
print(process_time)
start = time.time()
train_rev10 = augmentation.rotation_augment(train_rev, 10)
process_time = time.time() - start
print('train reverse rotate10 aug done')
print(process_time)
start = time.time()
train_rot5 = augmentation.rotation_augment(train, 5)
process_time = time.time() - start
print('train rotate5 aug done')
print(process_time)
start = time.time()
train_rot10 = augmentation.rotation_augment(train, 10)
process_time = time.time() - start
print('train rotate10 aug done')
print(process_time)
train = train.append(train_rev).append(train_rev5).append(train_rev10).append(train_rot5).append(train_rot10)
train.to_csv("train_aug.csv", header=False, index=False)
start = time.time()
train = preprocessing.preprocessing(train)
process_time = time.time() - start
train.to_csv("train_aug_normalized6.csv", header=False, index=False)
print('train preprocessing done')
print(process_time)

# test augmentation
start = time.time()
test_rev = augmentation.reverse_augment(test)
process_time = time.time() - start
print('test reverse aug done')
print(process_time)
test_rev5 = augmentation.rotation_augment(test_rev, 5)
process_time = time.time() - start
print('test reverse rootate5 aug done')
print(process_time)
start = time.time()
test_rev10 = augmentation.rotation_augment(test_rev, 10)
process_time = time.time() - start
print('test reverse rotate10 aug done')
print(process_time)
start = time.time()
test_rot5 = augmentation.rotation_augment(test, 5)
process_time = time.time() - start
print('test rotate5 aug done')
print(process_time)
start = time.time()
test_rot10 = augmentation.rotation_augment(test, 10)
process_time = time.time() - start
print('test rotate10 aug done')
print(process_time)
test = test.append(test_rev).append(test_rev5).append(test_rev10).append(test_rot5).append(test_rot10)
test.to_csv("test_aug.csv", header=False, index=False)
start = time.time()
test = preprocessing.preprocessing(test)
process_time = time.time() - start
test.to_csv("test_aug_normalized6.csv", header=False, index=False)
print('test preprocessing done')
print(process_time)
"""

X_train = train.loc[:, :column_len-2].values
y_train = train.loc[:, column_len-1].values

X_test = test.loc[:, :column_len-2].values
y_test = test.loc[:, column_len-1].values

num_classes = 3
#X_train = numpy.reshape(X_train, (X_train.shape[0], length_of_sequence, in_out_neurons))
#X_test = numpy.reshape(X_test, (X_test.shape[0], length_of_sequence, in_out_neurons))
#y_train = numpy.reshape(numpy.array(to_categorical(y_train)), [1, y_train.shape[0], 3])
#y_test = numpy.reshape(numpy.array(to_categorical(y_test)), [1, y_test.shape[0], 3])

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model(activation="relu"):
  model = Sequential([
#    Dense(500, input_shape=(500,)),
#    Dense(500), 
#    Dense(500), 
    Reshape((10, 50), input_shape=(500,)),
    TimeDistributed(Dense(n_hidden), input_shape=(length_of_sequence, in_out_neurons)),
    TimeDistributed(Dense(n_hidden)),
    #Reshape((10, 50)),
    LSTM(n_hidden, dropout=0.0, recurrent_dropout=0.0),
    #Flatten(),
    Dense(num_classes),
    #Flatten(),
    Activation('softmax')
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

  axR.set_ylabel('accuracy')
  axR.set_ylim([0.4,1.0])
  axR.legend(loc='lower right')

model = create_model()
model.summary()
es_cb = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')  
# Fit the model
start = time.time()
fit = model.fit(X_train, y_train, verbose=2, epochs=3000, validation_data=(X_test, y_test))#, callbacks=[es_cb])
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
fig.savefig("./image/s10_aug/d2_50_6.png")
