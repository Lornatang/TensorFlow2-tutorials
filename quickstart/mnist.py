#     Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""use tf.keras high-API"""

# To get started, import the TensorFlow library into your program:
import tensorflow as tf

# Load and prepare the MNIST dataset.
# Convert the samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Norm [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function used for training:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Train and evaluate model:
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

# The image classifier is now trained to ~98% accuracy on this dataset.

# ==============================================================================
# Epoch 1/10
# 60000/60000 [==============================] - 3s 50us/sample - loss: 0.3004 - accuracy: 0.9118
# Epoch 2/10
# 60000/60000 [==============================] - 3s 53us/sample - loss: 0.1453 - accuracy: 0.9569
# Epoch 3/10
# 60000/60000 [==============================] - 3s 46us/sample - loss: 0.1070 - accuracy: 0.9670
# Epoch 4/10
# 60000/60000 [==============================] - 3s 47us/sample - loss: 0.0897 - accuracy: 0.9718
# Epoch 5/10
# 60000/60000 [==============================] - 3s 47us/sample - loss: 0.0752 - accuracy: 0.9768
# Epoch 6/10
# 60000/60000 [==============================] - 3s 49us/sample - loss: 0.0645 - accuracy: 0.9792
# Epoch 7/10
# 60000/60000 [==============================] - 3s 51us/sample - loss: 0.0565 - accuracy: 0.9813
# Epoch 8/10
# 60000/60000 [==============================] - 3s 51us/sample - loss: 0.0523 - accuracy: 0.9836
# Epoch 9/10
# 60000/60000 [==============================] - 3s 49us/sample - loss: 0.0485 - accuracy: 0.9837
# Epoch 10/10
# 60000/60000 [==============================] - 3s 47us/sample - loss: 0.0454 - accuracy: 0.9851
# 10000/10000 [==============================] - 0s 35us/sample - loss: 0.0740 - accuracy: 0.9791
# ==============================================================================
