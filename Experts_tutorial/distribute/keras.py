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

# Setup
import tensorflow as tf
import tensorflow_datasets as tfds

import os

dataset, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = dataset['train'], dataset['test']

# Define distribution strategy
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# You can also do info.splits.total_num_examples to get the total
# number of examples in the dataset.
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


# resize and norm
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)


# build model
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.main = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu, input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

  def call(self, inputs, training=None, mask=None):
    with strategy.scope():
      outputs = self.main(inputs)

    return outputs


# Defile loss and optimizer
model = Model()

model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Define callbacks.
# Define the checkpoint directory to store the checkpoints

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif 3 <= epoch < 7:
    return 1e-4
  else:
    return 1e-5


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))


callbacks = [
  tf.keras.callbacks.TensorBoard(log_dir='./logs'),
  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                     save_weights_only=True),
  tf.keras.callbacks.LearningRateScheduler(decay),
  PrintLR()
]

# train and eval
model.fit(train_dataset, epochs=12, callbacks=callbacks)


model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)
print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


# export to saveModel
path = 'saved_model/'


tf.keras.experimental.export_saved_model(model, path)

unreplicated_model = tf.keras.experimental.load_from_saved_model(path)

unreplicated_model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer=tf.keras.optimizers.Adam(),
  metrics=['accuracy'])

eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


with strategy.scope():
  replicated_model = tf.keras.experimental.load_from_saved_model(path)
  replicated_model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

  eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
  print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
