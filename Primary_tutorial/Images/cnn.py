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
"""The most basic convolutional neural network is implemented by Tensorflow"""

# import TensorFlow and other library.
import tensorflow as tf
from tensorflow.python.keras import datasets,layers,models

# Download MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Create CNN model
class CNN(models.Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.feature = models.Sequential([
      layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
      layers.MaxPool2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
      layers.MaxPool2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation=tf.nn.relu)
    ])
    self.classifier = models.Sequential([
      layers.Flatten(),
      layers.Dense(64, activation=tf.nn.relu),
      layers.Dense(10, activation=tf.nn.softmax)
    ])

  def call(self, inputs, training=None, mask=None):
    x = self.feature(inputs)
    x = self.classifier(x)

    return x


# Load model and print model architecture.
model = CNN()
model.summary()

# Compile and train the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
