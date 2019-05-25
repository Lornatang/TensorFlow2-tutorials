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
"""This tutorial introduces word embeddings.
It contains complete code to train word embeddings from scratch on a small dataset,
and to visualize these embeddings using the Embedding Projector
(shown in the image below).
"""

# Import TensorFlow and other library
import tensorflow as tf
from tensorflow.python.keras import layers

import matplotlib.pyplot as plt

import io

# Keras makes it easy to use word embeddings. Let's take a look at the Embedding layer.
embedding_layer = layers.Embedding(1000, 32)

# We will train a sentiment classifier on IMDB movie reviews. In the process,
# we will learn embeddings from scratch.
# We will move quickly through the code that downloads and preprocesses
# the dataset (see this tutorial for more details).
vocab_size = 10000
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# Convert the integers back to words
# It may be useful to know how to convert integers back to text.
# Here, we'll create a helper function to query a dictionary object that contains
# the integer to string mapping.
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


# Movie reviews can be different lengths.
# We will use the pad_sequences function to standardize the lengths of the reviews.
maxlen = 500

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=maxlen)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                          value=word_index["<PAD>"],
                                                          padding='post',
                                                          maxlen=maxlen)

# Create a simple model
embedding_dim = 16


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.main = tf.keras.Sequential([
            layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(1, activation=tf.nn.sigmoid)
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.main(inputs)

        return x


# Load model
model = Model()
model.summary()

# Compile and train the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(
    train_data,
    train_labels,
    epochs=30,
    batch_size=512,
    validation_split=0.2)

# With this approach our model reaches a validation accuracy
# of around 88% (note the model is over-fitting,
# training accuracy is significantly higher).
history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5, 1))
plt.show()

# Next, let's retrieve the word embeddings learned during training.
# This will be a matrix of shape (vocab_size,embedding-dimension).
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)

# We will now write the weights to disk.
# To use the Embedding Projector,
# we will upload two files in tab separated format:
# a file of vectors (containing the embedding),
# and a file of meta data (containing the words).
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# If you are running this tutorial in Colaboratory,
# you can use the following snippet to download these files to
# your local machine (or use the file browser,
# View -> Table of contents -> File browser).
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download('vecs.tsv')
    files.download('meta.tsv')
