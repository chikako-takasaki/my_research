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
from sklearn.model_selection import GridSearchCV, GroupKFold
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib as mpl
import matplotlib.pyplot as plt
 
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

num_classes = 3
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

group_index = []
for i in range(1858):
  group_index = group_index + [i]*10

def create_model(activation="relu", nb_hidden=125, layer_num=4, dropout=0.2, bn='false'):
  model = Sequential([
    Dense(50, input_shape=(50,)),
    Activation(activation),
    Dropout(dropout),
  ])
  for i in range(layer_num) :
    model.add(Dense(nb_hidden))
  
    if bn == 'true' :
      model.add(BatchNormalization())
    
    model.add(Activation(activation))
    model.add(Dropout(dropout))
   
  model.add(Dense(3))
  model.add(Activation('sigmoid'))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  return model

model = KerasClassifier(build_fn=create_model, verbose=0)
middle_layer_node_num = [50, 75, 100, 125]
middle_layer_num = [3, 4, 5, 6]
dropout = [0.0, 0.2, 0.3, 0.4, 0.5]
bn = ['true', 'false']
epoch = [800, 1200, 1600, 2000]
param_grid = dict(dropout=dropout, bn=bn)
gkf = GroupKFold(n_splits=3).split(X_train,y_train,group_index)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=gkf)


#model = create_model()
#model.summary()
start = time.time()
#fit = model.fit(X_train, y_train, verbose=2, epochs=2000, validation_data=(X_test, y_test))
grid_result = grid.fit(X_train, y_train, verbose=2, epochs=1600, validation_data=(X_test, y_test))
process_time = time.time() - start

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

scores = results.pivot("param_bn", "param_dropout", "mean_test_score")
print(scores)

plt.figure()
sns.heatmap(scores, annot=True)
plt.savefig('image/allv2_dobn_heatmap_1600.png')

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
