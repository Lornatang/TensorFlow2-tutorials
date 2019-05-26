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

"""This guide uses machine learning to categorize Iris flowers by species.
It uses TensorFlow to: 1. Build a model, 2. Train this model on example data, and 3.
Use the model to make predictions about unknown data.
"""

# Import TensorFlow and other library
import tensorflow as tf

import matplotlib.pyplot as plt

import os

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Import and parse the training dataset
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename('./iris_training.csv'),
                                           origin=train_dataset_url)

# train_dataset_fp = tf.keras.utils.get_file
print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

"""
Features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] 
Label: species
"""

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Create a tf.data.Dataset
batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
  train_dataset_fp,
  batch_size,
  column_names=column_names,
  label_name=label_name,
  num_epochs=1)

features, labels = next(iter(train_dataset))

print(features)

# You can start to see some clusters by plotting a few features from the batch:
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()


# This function uses the tf.stack method which takes values from a list of tensors
# and creates a combined tensor at the specified dimension.
def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


# Then use the tf.data.Dataset.map method to pack the features of each
# (features,label) pair into the training dataset:
train_dataset = train_dataset.map(pack_features_vector)

# The features element of the Dataset are now arrays with shape
# (batch_size, num_features). Let's look at the first few examples:
features, labels = next(iter(train_dataset))

print(features[:5])

# Create a model using Keras
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

# Using the model
predictions = model(features)

print(predictions[:5])

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

"""
Prediction: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    Labels: [1 2 0 0 1 2 2 0 0 2 0 2 1 1 1 0 0 0 2 1 2 0 0 1 1 2 1 1 0 2 2 0]
"""

# Train the model
# Define the loss and gradient function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))

"""
Loss test: 2.3108744621276855
"""


# Use the tf.GradientTape context to calculate the gradients used to optimize
# our model.
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# We'll use this to calculate a single optimization step.
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels).numpy()))

"""
Step: 0, Initial Loss: 2.3108744621276855
Step: 1,         Loss: 1.7618987560272217
"""

# Training loop
# Note: Rerunning this cell uses the same model variables
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(y, model(x))

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

"""
Epoch 000: Loss: 1.568, Accuracy: 30.000%
Epoch 050: Loss: 0.061, Accuracy: 98.333%
Epoch 100: Loss: 0.058, Accuracy: 97.500%
Epoch 150: Loss: 0.044, Accuracy: 99.167%
Epoch 200: Loss: 0.049, Accuracy: 97.500%
"""

# Visualize the loss function over time
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# Setup the test dataset
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.data.experimental.make_csv_dataset(
  test_fp,
  batch_size,
  column_names=column_names,
  label_name='species',
  num_epochs=1,
  shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

# Evaluate the model on the test dataset
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# We can see on the last batch, for example, the model is usually correct.
tf.stack([y, prediction], axis=1)

# Use the trained model to make predictions
predict_dataset = tf.convert_to_tensor([
  [5.1, 3.3, 1.7, 0.5, ],
  [5.9, 3.0, 4.2, 1.5, ],
  [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print(f"Example {i} prediction: {name} ({100 * p:4.1f}%)")

"""
Example 0 prediction: Iris setosa (100.0%)
Example 1 prediction: Iris versicolor (100.0%)
Example 2 prediction: Iris virginica (99.5%)
"""
