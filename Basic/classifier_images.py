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
"""This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
   We will use 60000 images to train the network and 10000 images to evaluate how accurately the network learned to
   classify images.Uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories.
   The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# define EPOCHS
EPOCHS = 5

# Import the Fashion MNIST dataset.
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Use tf.data to batch and shuffle the dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64)

# Each image is mapped to a single label.
# Since the class names are not included with the dataset, store them here to use later when plotting the images.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data
# If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255.
train_images, test_images = train_images / 255.0, test_images / 255.0


# Build the model
# Building the neural network requires configuring the layers of the model, then compiling the model.
class CNN(keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    # The first layer in this network,'tf.keras.layers.Flatten',
    # transforms the format of the images from
    # a two-dimensional array (of 28 by 28 pixels)
    # to a one-dimensional array (of 28 * 28 = 784 pixels).
    # Think of this layer as unstacking rows of pixels in the image and lining them up.
    # This layer has no parameters to learn; it only reformats the data.
    self.f1 = keras.layers.Flatten(input_shape=(28, 28))
    # After the pixels are flattened, the network consists of a sequence of two
    # 'tf.keras.layers.Dense layers'.
    # These are densely connected, or fully connected, neural layers.
    # The first Dense layer has 128 nodes (or neurons).
    # The second (and last) layer is a 10-node softmax layer that returns
    # an array of 10 probability scores that sum to1.
    # Each node contains a score that indicates the probability that the
    # current image belongs to one of the 10 classes.
    self.d1 = keras.layers.Dense(128, activation=tf.nn.relu)
    self.d2 = keras.layers.Dense(10, activation=tf.nn.softmax)

  def call(self, inputs, training=None, mask=None):
    inputs = self.f1(inputs)  # forward flatten
    inputs = self.d1(inputs)  # forward dense1
    outputs = self.d2(inputs)  # forward dense2 and output
    return outputs


model = CNN()

# Loss function
# This measures how accurate the model is during training.
# We want to minimize this function to "steer" the model in the right direction.
entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Optimizer.
# This is how the model is updated based on the data it sees and its loss function.
optimizer = tf.keras.optimizers.Adam()

# Metrics.
# Used to monitor the training and testing steps.
# The following example uses accuracy, the fraction of the images that are correctly classified.
metrics = tf.keras.metrics.Accuracy()


# Use tf.GradientTape to train the model.
@tf.function
def train_step(images, labels):
  """ batch size dataset for train.

  Args:
    images: train images.
    labels: train labels.

  Returns:
    loss, accuracy

  """
  with tf.GradientTape() as tape:
    predictions = model(images)

  loss = entropy(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  loss = entropy(labels, predictions)
  accuracy = metrics(labels, predictions)

  return loss, accuracy


# Test the model.
@tf.function
def test_step(images, labels):
  """ batch size dataset for train.

  Args:
    images: test images.
    labels: test labels.

  Returns:
    loss, accuracy

  """
  predictions = model(images)

  loss = entropy(labels, predictions)
  accuracy = metrics(labels, predictions)

  return loss, accuracy


# training
def train():
  for epoch in range(EPOCHS):
    # init all paras
    loss, accuracy, test_loss, test_accuracy = 0., 0., 0., 0.
    for images, labels in train_dataset:
      loss, accuracy = train_step(images, labels)

    for test_images, test_labels in test_dataset:
      test_loss, test_accuracy = test_step(test_images, test_labels)

    print(f"Epoch {epoch + 1},"
          f"Loss: {loss.result() * 100:.6f},"
          f"Accuracy: {accuracy.result():.4f}%,"
          f"Test Loss: {test_loss.result() * 100:.6f},"
          f"test Accuracy: {test_accuracy.result():.4f}%.")


# prediction
# With the model trained, we can use it to make predictions about some images.
def pred():
  predictions = model.predict(test_images)
  print(f"This images classifier is {np.argmax(predictions[0])}")


# So, the model is most confident that this image is an ankle boot.
# Examining the test label shows that this classification is correct.
def plot_image(i, predictions_array, true_label, img):
  """ plot images.

  Args:
    i: nums of dataset.
    predictions_array: pred array [0., 0., 0., 0., 0., 0., 0., 9., 1.].
    true_label: real label.
    img: input images data.

  """
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.gray())

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                       100 * np.max(predictions_array),
                                       class_names[true_label]),
             color=color)


def plot_value_array(i, predictions_array, true_label):
  """ plt classifier value.

  Args:
    i: nums of dataset.
    predictions_array: pred array [0., 0., 0., 0., 0., 0., 0., 9., 1.].
    true_label: real label.

  """
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# main func
if __name__ == '__main__':
  train()
  # pred and plot
  predictions = model.predict(test_images[0])
  plot_value_array(0, predictions, test_labels)
  _ = plt.xticks(range(10), class_names, rotation=45)
