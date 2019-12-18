import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import  BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

import pandas as pd
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
 
data = pd.read_csv('~/work/s10_100_2pre.csv', header=None)
column_len = len(data.columns)

X = data.loc[:, :column_len-2].values
y = data.loc[:, column_len-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model(activation="relu", nb_hidden=500, layer_num=3, dropout=0.2):
  model = Sequential([
    Dense(500, input_shape=(500,)),
    Activation(activation),
    Dropout(dropout),
  ])
  for i in range(layer_num) :
    model.add(Dense(nb_hidden))
#    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'top_k_categorical_accuracy'])
  return model

#model = KerasClassifier(build_fn=create_model, verbose=0)
#param_grid = dict(nb_epoch=[10, 20, 25], batch_size=[5, 10, 100, 200])
#param_grid = dict(dropout=[0.2, 0.3, 0.4, 0.5])
#grid = GridSearchCV(estimator=model, param_grid=param_grid)

model = create_model(layer_num=3, nb_hidden=500)
model.summary()
start = time.time()
fit = model.fit(X_train, y_train, verbose=2, epochs=500, validation_data=(X_test, y_test))
process_time = time.time() - start

#start = time.time()
#grid_result = grid.fit(X_train, y_train, verbose=2, epochs=800, validation_data=(X_test, y_test))
#process_time = time.time() - start
#print (grid_result.best_score_)
#print (grid_result.best_params_)

print("training time")
print(process_time)

train_score = model.evaluate(X_train, y_train, verbose=0)
print("train loss:",train_score[0])
print('train accuracy : ', train_score[1])

start = time.time()
test_score = model.evaluate(X_test, y_test, verbose=0)
process_time = time.time() - start
print("Test loss:", test_score[0])
print('test accuracy : ', test_score[1])
print("model evaluate time")
print(process_time)

print('top5 accurarcy')
print('train : ',train_score[2])
print('test : ',test_score[2])
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
def plot_history_loss(fit):
  # Plot the loss in the history
  axL.plot(fit.history['loss'],label="loss for training")
  axL.plot(fit.history['val_loss'],label="loss for validation")
  axL.set_title('model loss')
  axL.set_xlabel('epoch')
  axL.set_ylabel('loss')
  axL.legend(loc='upper right')

def plot_history_acc(fit):
  # Plot the loss in the history
  axR.plot(fit.history['top_k_categorical_accuracy'],label="accuracy for training")
  axR.plot(fit.history['val_top_k_categorical_accuracy'],label="accuracy for validation")
  axR.set_title('model accuracy')
  axR.set_xlabel('epoch')
  axR.set_ylabel('accuracy')
  axR.legend(loc='lower right')


plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig('./image/100ctg_v2/s10/all3dr2_500.png')
plt.close()
