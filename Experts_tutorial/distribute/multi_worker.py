# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This tutorial demonstrates how to distribute a tf.distribute.Strategy with
   a distributed multi-worker training tf.estimator.
   If you are using the code tf.estimator, and you want to extend it to
   a single machine with high performance, this tutorial is for you.
"""

import json
import os

# Setup
import tensorflow as tf
import tensorflow_datasets as tfds

# Input function
BUFFER_SIZE = 10000
BATCH_SIZE = 64


def input_fn(mode, input_context=None):
  datasets, info = tfds.load(name='mnist',
                             with_info=True,
                             as_supervised=True)
  mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  if input_context:
    mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
  return mnist_dataset.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


NUM_WORKERS = 1
IP_ADDRS = ['localhost']
PORTS = [12345]

os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
    'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)]
  },
  'task': {'type': 'worker', 'index': 0}
})

# Build model
LEARNING_RATE = 1e-4


def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  logits = model(features, training=False)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=LEARNING_RATE)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)(labels, logits)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=optimizer.minimize(
      loss, tf.compat.v1.train.get_or_create_global_step()))


# MultiWorkerMirroredStrategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# train and evaluate
config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
  model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
  classifier,
  train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
  eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)

### END TUTORIALS
#### THANKS READ MY CODE
