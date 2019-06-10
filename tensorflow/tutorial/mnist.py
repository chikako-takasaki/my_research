import input_data
import numpy as np
import tensorflow as tf
from sklearn import datasets

def inference(x, keep_prob, n_in , n_hiddens, n_out):
  # モデルの定義
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

  # 入力層 - 隠れ層、隠れ層 - 隠れ層
  for i, n_hidden in enunerate(n_hiddens):
    if i == 0:
      input = x
      input_dim = n_in
    else:
      input = output
      input_dim = n_hiddens[i-1]

    W = weight_variable([input_dim, n_hidden])
    b = bias_variable([n_hidden])

    h = tf.nn.reru(tf.matmul(input, W) + b)
    output = tf.nn.dropout(h, keep_prob)

  # 隠れ層 - 出力層
  W_out = weight_variable([n_hiddens[-1], n_out])
  b_out = bias_variable([n_out])
  y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)
  return y

def loss(y, t):
  cross_entropy = tf.reduce_mean(
    -tf.reduce_sun(t * tf.log(y), reduction_indices=[1]))
  return cross_entropy

def training(ross):
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train_step = optimizer.minimize(loss)
  return train_step

if __name__ == '__main__':
  # 1. データの準備
  mnist = datasets.fetch_mldata('MNIST original', data_home='.')

  n = len(mnist.data)
  N = 10000
  indices = np.random.permutation(range(n))[:N]
  X = mnist.data[indices]
  y = mnist.target[indices]
  Y = np.eye(10)[y.astype(int)]

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

 # 2. モデル設定
  n_in = len(X[10])
  n_hiddens = [200, 200, 200]
  n_out = len(Y[0])

  x = tf.placeholder(tf.float32, shape=[None, n_in])
  keep_prob = tf.placeholder(tf.float32)

  y = inference(x, keep_prob, n_in=n_in, n_hiddens=n_hiddens, n_out=n_out)
