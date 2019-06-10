import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /=255

num_classes = 10
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential([
  Dense(512, activation='relu', input_shape=(784,)),
  Dense(512, activation='relu'),
  Dense(10, activation='sigmoid')
])
  
model.summary()

batch_size = 128
epochs = 20

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

training_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)

print('Training loss:', training_score[0])
print('Training accuracy:', training_score[1])

print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
