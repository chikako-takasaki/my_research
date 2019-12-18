# -*- coding: utf-8 -*-
# Reference URL: https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import numpy as np
import json

def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
  graph = sess.graph
  with graph.as_default():
    freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
    output_names = output_names or []
    output_names += [v.op.name for v in tf.global_variables()]
    input_graph_def = graph.as_graph_def()
    if clear_devices:
      for node in input_graph_def.node:
        node.device = ''
    frozen_graph = convert_variables_to_constants(sess, input_graph_def, output_names, freeze_var_names)
    return frozen_graph

def convert_keras_to_tensorflow(keras_model_filename, tf_model_filename):
  model = load_model(keras_model_filename)
  model.summary()
  frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
  tf.train.write_graph(frozen_graph, './', tf_model_filename, as_text=False)

def get_model_info(tf_model_filename):
  ops = {}
  with tf.Session() as sess:
    with tf.gfile.GFile(tf_model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      _ = tf.import_graph_def(graph_def)
      for op in tf.get_default_graph().get_operations():
        print (op.name)
        ops[op.name] = [str(output) for output in op.outputs]
        for output in op.outputs:
          print(output)
    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()

  with open(tf_model_filename+'_param.json', 'w') as f:
    f.write(json.dumps(ops))


if __name__ == '__main__':
  # convert
  keras_model_filename = 'mnist_dense.h5'
  tf_model_filename = 'mnist_dense.pb'
  convert_keras_to_tensorflow(keras_model_filename, tf_model_filename)
  get_model_info(tf_model_filename)
