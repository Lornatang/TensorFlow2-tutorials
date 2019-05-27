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

import tensorflow as tf

import numpy as np

"""## Introduction to Keras callbacks
In Keras, `Callback` is a python class meant to be subclassed to provide specific functionality, with a set of methods called at various stages of training (including batch/epoch start and ends), testing, and predicting. Callbacks are useful to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument `callbacks`) to any of `tf.keras.Model.fit()`, `tf.keras.Model.evaluate()`, and `tf.keras.Model.predict()` methods. The methods of the callbacks will then be called at different stages of training/evaluating/inference.

To get started, let's import tensorflow and define a simple Sequential Keras model:
"""


# Define the Keras model to add callbacks to
def get_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(1, activation='linear', input_dim=784))
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])
  return model


"""Then, load the MNIST data for training and testing from Keras datasets API:"""

# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

"""Now, define a simple custom callback to track the start and end of every batch of data. During those calls, it prints the index of the current batch."""

import datetime


class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


"""Providing a callback to model methods such as `tf.keras.Model.fit()` ensures the methods are called at those stages:"""

model = get_model()
_ = model.fit(x_train, y_train,
              batch_size=64,
              epochs=1,
              steps_per_epoch=5,
              verbose=0,
              callbacks=[MyCustomCallback()])

"""## Model methods that take callbacks
Users can supply a list of callbacks to the following `tf.keras.Model` methods:
#### [`fit()`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit), [`fit_generator()`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit_generator)
Trains the model for a fixed number of epochs (iterations over a dataset, or data yielded batch-by-batch by a Python generator).
#### [`evaluate()`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#evaluate), [`evaluate_generator()`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#evaluate_generator)
Evaluates the model for given data or data generator. Outputs the loss and metric values from the evaluation.
#### [`predict()`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#predict), [`predict_generator()`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#predict_generator)
Generates output predictions for the input data or data generator.
"""

_ = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=5,
                   callbacks=[MyCustomCallback()])

"""## An overview of callback methods


### Common methods for training/testing/predicting
For training, testing, and predicting, following methods are provided to be overridden.
#### `on_(train|test|predict)_begin(self, logs=None)`
Called at the beginning of `fit`/`evaluate`/`predict`.
#### `on_(train|test|predict)_end(self, logs=None)`
Called at the end of `fit`/`evaluate`/`predict`.
#### `on_(train|test|predict)_batch_begin(self, batch, logs=None)`
Called right before processing a batch during training/testing/predicting. Within this method, `logs` is a dict with `batch` and `size` available keys, representing the current batch number and the size of the batch.
#### `on_(train|test|predict)_batch_end(self, batch, logs=None)`
Called at the end of training/testing/predicting a batch. Within this method, `logs` is a dict containing the stateful metrics result.

### Training specific methods
In addition, for training, following are provided.
#### on_epoch_begin(self, epoch, logs=None)
Called at the beginning of an epoch during training.
#### on_epoch_end(self, epoch, logs=None)
Called at the end of an epoch during training.

### Usage of `logs` dict
The `logs` dict contains the loss value, and all the metrics at the end of a batch or epoch. Example includes the loss and mean absolute error.
"""


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

  def on_train_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  def on_test_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  def on_epoch_end(self, epoch, logs=None):
    print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'],
                                                                                                logs['mae']))


model = get_model()
_ = model.fit(x_train, y_train,
              batch_size=64,
              steps_per_epoch=5,
              epochs=3,
              verbose=0,
              callbacks=[LossAndErrorPrintingCallback()])

"""Similarly, one can provide callbacks in `evaluate()` calls."""

_ = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=20,
                   callbacks=[LossAndErrorPrintingCallback()])

"""## Examples of Keras callback applications
The following section will guide you through creating simple Callback applications.

### Early stopping at minimum loss
First example showcases the creation of a `Callback` that stops the Keras training when the minimum of loss has been reached by mutating the attribute `model.stop_training` (boolean). Optionally, the user can provide an argument `patience` to specfify how many epochs the training should wait before it eventually stops.

`tf.keras.callbacks.EarlyStopping` provides a more complete and general implementation.
"""


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('loss')
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


model = get_model()
_ = model.fit(x_train, y_train,
              batch_size=64,
              steps_per_epoch=5,
              epochs=30,
              verbose=0,
              callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])

"""### Learning rate scheduling
One thing that is commonly done in model training is changing the learning rate as more epochs have passed. Keras backend exposes get_value api which can be used to set the variables. In this example, we're showing how a custom Callback can be used to dymanically change the learning rate.

`tf.keras.callbacks.LearningRateScheduler` provides a more general implementation.
"""


class LearningRateScheduler(tf.keras.callbacks.Callback):
  """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

  def __init__(self, schedule):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))


LR_SCHEDULE = [
  # (epoch to start, learning rate) tuples
  (3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)
]


def lr_schedule(epoch, lr):
  """Helper function to retrieve the scheduled learning rate based on epoch."""
  if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
    return lr
  for i in range(len(LR_SCHEDULE)):
    if epoch == LR_SCHEDULE[i][0]:
      return LR_SCHEDULE[i][1]
  return lr


model = get_model()
_ = model.fit(x_train, y_train,
              batch_size=64,
              steps_per_epoch=5,
              epochs=15,
              verbose=0,
              callbacks=[LossAndErrorPrintingCallback(), LearningRateScheduler(lr_schedule)])

"""### Standard Keras callbacks
Be sure to check out the existing Keras callbacks by [visiting the api doc](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/callbacks). Applications include logging to CSV, saving the model, visualizing on TensorBoard and a lot more.
"""
