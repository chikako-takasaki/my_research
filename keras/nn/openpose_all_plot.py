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
 
data = pd.read_csv('openpose_data_all.csv')
X = data.drop("category", axis=1)
y = data["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train = X_train.values
X_train /= 1920.0
X_test = X_test.values
X_test /= 1920.0
y_train = y_train.values
y_test = y_test.values

num_classes = 3
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model(activation="relu", nb_hidden=150, layer_num=4):
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
   
  model.add(Dense(3))
  model.add(Activation('sigmoid'))

  model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
  return model

model3 = create_model(layer_num=3)
model3.summary()
start3 = time.time()
fit3 = model3.fit(X_train, y_train, verbose=2, epochs=800, validation_data=(X_test, y_test))
process_time3 = time.time() - start3

model4 = create_model(layer_num=4)
model4.summary()
start4 = time.time()
fit4 = model4.fit(X_train, y_train, verbose=2, epochs=800, validation_data=(X_test, y_test))
process_time4 = time.time() - start4

model5 = create_model(layer_num=5)
model5.summary()
start5 = time.time()
fit5 = model5.fit(X_train, y_train, verbose=2, epochs=800, validation_data=(X_test, y_test))
process_time5 = time.time() - start5

model6 = create_model(layer_num=6)
model6.summary()
start6 = time.time()
fit6 = model6.fit(X_train, y_train, verbose=2, epochs=800, validation_data=(X_test, y_test))
process_time6 = time.time() - start6

#start = time.time()
#grid_result = grid.fit(X_train, y_train, verbose=2, epochs=800, validation_data=(X_test, y_test))
#process_time = time.time() - start
#print (grid_result.best_score_)
#print (grid_result.best_params_)

print("---------- middle layer num : 3 ------------")
print("model3 training time")
print(process_time3)

train_score = model3.evaluate(X_train, y_train, verbose=0)
print("train loss:",train_score[0])
print('train accuracy : ', train_score[1])

test_score = model3.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_score[0])
print('test accuracy : ', test_score[1])


print("---------- middle layer num : 4 ------------")
print("model4 training time")
print(process_time4)

train_score = model4.evaluate(X_train, y_train, verbose=0)
print("train loss:",train_score[0])
print('train accuracy : ', train_score[1])

test_score = model4.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_score[0])
print('test accuracy : ', test_score[1])


print("---------- middle layer num : 5 ------------")
print("model5 training time")
print(process_time5)

rain_score = model5.evaluate(X_train, y_train, verbose=0)
print("train loss:",train_score[0])
print('train accuracy : ', train_score[1])

test_score = model5.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_score[0])
print('test accuracy : ', test_score[1])


print("---------- middle layer num : 6 ------------")
print("model6 training time")
print(process_time6)

train_score = model6.evaluate(X_train, y_train, verbose=0)
print("train loss:",train_score[0])
print('train accuracy : ', train_score[1])

test_score = model6.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_score[0])
print('test accuracy : ', test_score[1])


# plot 
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
def plot_history_loss(fit3, fit4, fit5, fit6):
  # Plot the loss in the history
  axL.plot(fit3.history['val_loss'],label="Middle layer num : 3")
  axL.plot(fit4.history['val_loss'],label="Middle layer num : 4")
  axL.plot(fit5.history['val_loss'],label="Middle layer num : 5")
  axL.plot(fit6.history['val_loss'],label="Middle layer num : 6")
  axL.set_title('model loss')
  axL.set_xlabel('epoch')
  axL.set_ylabel('loss')
  axL.set_ylim([0.6, 1.0])
  axL.legend(loc='lower right')

def plot_history_acc(fit3, fit4, fit5, fit6):
  # Plot the loss in the history
  axR.plot(fit3.history['val_acc'],label="Middle layer num : 3")
  axR.plot(fit4.history['val_acc'],label="Middle layer num : 4")
  axR.plot(fit5.history['val_acc'],label="Middle layer num : 5")
  axR.plot(fit6.history['val_acc'],label="Middle layer num : 6")
  axR.set_title('model accuracy')
  axR.set_xlabel('epoch')
  axR.set_ylabel('accuracy')
  axR.set_ylim([0.9, 1.0])
  axR.legend(loc='lower right')


plot_history_loss(fit3, fit4, fit5, fit6)
plot_history_acc(fit3, fit4, fit5, fit6)
fig.savefig('./image/all_plot09_800.png')
plt.close()
