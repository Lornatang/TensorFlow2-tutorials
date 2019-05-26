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
"""The tutorial demonstrates the basic application of transfer learning with TensorFlow Hub and Keras.
   We'll use the IMDB dataset that contains the text of 50,000 movie reviews from the Internet Movie Database.
   These are split into 25,000 reviews for training and 25,000 reviews for testing.
   The training and testing sets are balanced, meaning they contain an equal number of positive and negative reviews.
"""

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# print some information
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# Download the IMDB dataset.
# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
  name="imdb_reviews",
  split=(train_validation_split, tfds.Split.TEST),
  as_supervised=True)

# Explore the data
# Let's take a moment to understand the format of the data.
# Each example is a sentence representing the movie review and a corresponding label.
# The sentence is not preprocessed in any way.
# The label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# Build the model
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# Let's now build the full model.
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# Loss function and optimizer.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model.
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))