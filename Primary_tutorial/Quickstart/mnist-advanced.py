#     Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
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

# Import TensorFlow into your program:
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras import Model

# define epochs
EPOCHS = 5

# Load and prepare the MNIST dataset.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Use tf.data to batch and shuffle the dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)


# Build the tf.keras model using the Keras model subclassing API:
class CNN(Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = Conv2D(32, 3, activation=tf.nn.relu)
    self.flatten = Flatten()
    self.d1 = Dense(128, activation=tf.nn.relu)
    self.d2 = Dense(10, activation=tf.nn.softmax)

  def call(self, x, **kwargs):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


model = CNN()

# Choose an optimizer and loss function for training.
loss_op = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model.
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Use tf.GradientTape to train the model.
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_op(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


# Test the model.
@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_op(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


# training
def train():
  for epoch in range(EPOCHS):
    for images, labels in train_dataset:
      train_step(images, labels)

    for test_images, test_labels in test_dataset:
      test_step(test_images, test_labels)

    print(f"Epoch {epoch+1},"
          f"Loss: {train_loss.result() * 100:.6f},"
          f"Accuracy: {train_accuracy.result():.4f}%,"
          f"Test Loss: {test_loss.result() * 100:.6f},"
          f"test Accuracy: {test_accuracy.result():.4f}%.")


# main func
if __name__ == '__main__':
    train()


# The image classifier is now trained to ~98% accuracy on this dataset.

# ==============================================================================
# Epoch 1, Loss: 0.151985, Accuracy: 95.5033%,
# Test Loss: 0.073297, Test Accuracy: 97.6999%.

# Epoch 2, Loss: 0.097963, Accuracy: 97.0883%,
# Test Loss: 0.065212, Test Accuracy: 97.9150%.

# Epoch 3, Loss: 0.072738, Accuracy: 97.8533%,
# Test Loss: 0.063016, Test Accuracy: 97.9833%.

# Epoch 4, Loss: 0.057954, Accuracy: 98.2820%,
# Test Loss: 0.061889, Test Accuracy: 98.0650%.

# Epoch 5, Loss: 0.048282, Accuracy: 98.5673%,,
# Test Loss: 0.061678, Test Accuracy: 98.1159%.
# ==============================================================================
