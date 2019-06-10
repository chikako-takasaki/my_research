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
 
data = pd.read_csv('openpose_data3.csv')
X = data.drop("category", axis=1)
y = data["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train = X_train.values
X_train /= 1920.0
X_test = X_test.values
X_test /= 1920.0
y_train = y_train.values
y_test = y_test.values

num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model(activation="relu", nb_hidden=100, layer_num=5):
  model = Sequential([
    Dense(50, input_shape=(50,)),
    Activation(activation),
#    Dropout(dropout),
  ])
  for i in range(layer_num) :
    model.add(Dense(nb_hidden))
#   model.add(BatchNormalization())
    model.add(Activation(activation))
#   model.add(Dropout(dropout))
   
  model.add(Dense(2))
  model.add(Activation('sigmoid'))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  return model

#model = KerasClassifier(build_fn=create_model, verbose=0)
#param_grid = dict(nb_epoch=[10, 20, 25], batch_size=[5, 10, 100, 200])
#param_grid = dict(dropout=[0.2, 0.3, 0.4, 0.5])
#grid = GridSearchCV(estimator=model, param_grid=param_grid)

model = create_model()
model.summary()
start = time.time()
fit = model.fit(X_train, y_train, verbose=2, epochs=2000, validation_data=(X_test, y_test))
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
fig.savefig('./image/rw5_2000.png')
plt.close()
