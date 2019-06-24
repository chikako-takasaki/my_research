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

# パラメーター
length_of_sequence = 10 
in_out_neurons = 50
n_hidden = 50

np.set_printoptions(suppress=True)
def preprocessing(data):
  data_len = len(data)
  for i in range(data_len):
    data_row = data.iloc[i]
    max_v = max(data_row)
    min_v = min(data_row)
    for c in range(data_row.shape[0]):
      v = data_row.values[c]
      new_v = (v - min_v) / (max_v - min_v)
      data.iloc[i,c] = new_v
  return data.values

# データ整形
data = pd.read_csv('~/work/normalized_s10.csv', header=None)
column_len = len(data.columns)
X = data.loc[:, 1 : (column_len-2)]
y = data.loc[:, column_len-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_test2 = X_test
#X_test = numpy.array(X_test)
#zeros  = numpy.zeros((X_train.shape[0] - X_test.shape[0], X_train.shape[1]))
#X_test = numpy.append(X_test, zeros, axis = 0)
#X_test = pd.DataFrame(X_test)

X_train = X_train.values
#X_train /= 1920.0
X_test = X_test.values
#X_test /= 1920.0
#X_train = preprocessing(X_train)
#X_test = preprocessing(X_test)
y_train = y_train.values
y_test = y_test.values

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
fig.savefig("./image/s10_v2/d2_50_2.png")

# pick up false data
pred = model.predict_classes(X_test)
for i, (p, y) in enumerate(zip(pred, y_test)):
  y = np.where(y == 1)[0][0]
  if p != y:
    print('data_num: {}, predict class: {}, correct class: {}'.format(X_test2.index[i], p, y))
