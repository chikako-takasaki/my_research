import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import  BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib as mpl
import matplotlib.pyplot as plt
 
data = pd.read_csv('~/work/s10_100_pre.csv', header=None)
column_len = len(data.columns)
X = data.loc[:, 0 : (column_len-2)]
y = data.loc[:, column_len-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

num_classes = 100
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model(activation="relu", nb_hidden=700, layer_num=4, dropout=0.0, bn='false'):
  model = Sequential([
    Dense(500, input_shape=(500,)),
    Activation(activation),
    Dropout(dropout),
  ])
  for i in range(layer_num) :
    model.add(Dense(nb_hidden))
  
    if bn == 'true' :
      model.add(BatchNormalization())
    
    model.add(Activation(activation))
    model.add(Dropout(dropout))
   
  model.add(Dense(num_classes))
  model.add(Activation('sigmoid'))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy','top_k_categorical_accuracy'])
  return model

model = KerasClassifier(build_fn=create_model, verbose=0)
middle_layer_node_num = [300, 400, 500, 600, 700]
#middle_layer_node_num = [300, 400, 500, 600, 700]
middle_layer_num = [1, 2, 3, 4, 5]
#middle_layer_num = [3, 4, 5, 6]
dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
bn = ['true', 'false']
#param_grid = dict(dropout=dropout, bn=bn)
param_grid = dict(nb_hidden=middle_layer_node_num, layer_num=middle_layer_num)
scoring = ['accuracy', 'top_k_categorical_accuracy']
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='top_k_categorical_accuracy', cv=2)

#model = create_model()
#model.summary()
start = time.time()
#fit = model.fit(X_train, y_train, verbose=2, epochs=2000, validation_data=(X_test, y_test))
grid_result = grid.fit(X_train, y_train, verbose=2, epochs=1, validation_data=(X_test, y_test))
process_time = time.time() - start

#gs_result = pd.DataFrame.from_dict(grid.cv_results_)
#gs_result.to_csv('100ctg_100_grid.csv')
#start = time.time()
#process_time = time.time() - start
print ("best_score : ", grid_result.best_score_)
print ("best_params : ", grid_result.best_params_)

print("training time")
print(process_time)

results = pd.DataFrame(grid_result.cv_results_)
print("shape: ", results.shape)
print(results.columns)

print(results)
print(gs_result)
#scores = results.pivot("param_bn", "param_dropout", "mean_test_score")
#plt.figure()
#sns.heatmap(scores, annot=True)
#plt.savefig('image/stream_heatmap_dobn1600.png')
"""
train_score = model.evaluate(X_train, y_train, verbose=0)
print("train loss:",train_score[0])
print('train accuracy : ', train_score[1])

test_score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_score[0])
print('test accuracy : ', test_score[1])

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
  axR.plot(fit.history['acc'],label="accuracy for training")
  axR.plot(fit.history['val_acc'],label="accuracy for validation")
  axR.set_title('model accuracy')
  axR.set_xlabel('epoch')
  axR.set_ylabel('accuracy')
  axR.legend(loc='upper right')


plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig('./image/all3_dobn_2000.png')
plt.close()
"""
