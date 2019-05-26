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
"""This notebook classifies movie reviews as positive or negative using the text
   of the review. This is an example of binary—or two-class—classification,
   an important and widely applicable kind of machine learning problem.

   We'll use the IMDB dataset that contains the text of 50,000 movie reviews
   from the Internet Movie Database. These are split into 25,000 reviews for
   training and 25,000 reviews for testing.
   The training and testing sets are balanced,
   meaning they contain an equal number of positive and negative reviews.

   This notebook uses tf.keras, a high-level API to build and train models
   in TensorFlow. For a more advanced text classification tutorial
   using tf.keras, see the MLCC Text Classification Guide.
"""

# Import TensorFlow and other library
import tensorflow as tf
from tensorflow.python import keras

import matplotlib.pyplot as plt

# Download the IMDB dataset
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore the data
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# Convert the integers back to words
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])


# Prepare the data
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# Build the model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000


class Model(keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.main = keras.Sequential([
      keras.layers.Embedding(vocab_size, 16),
      keras.layers.GlobalAveragePooling1D(),
      keras.layers.Dense(16, activation=tf.nn.relu),
      keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

  def call(self, inputs, training=None, mask=None):
    x = self.main(inputs)

    return x


# Load model
model = Model()
model.summary()

# Loss function and optimizer
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

# Create a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
