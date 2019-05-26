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
"""This text classification tutorial trains a recurrent neural network on the
   IMDB large movie review dataset for sentiment analysis.
"""

# Import TensorFlow and other library
import tensorflow as tf
from tensorflow.python import keras

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt


# Create a helper function to plot graphs.
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# Setup input pipeline
# Download the dataset using TFDS. The dataset comes with an inbuilt subword tokenizer.
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# As this is a subwords tokenizer, it can be passed any string and the tokenizer will
# tokenize it.
tokenizer = info.features['text'].encoder
print('Vocabulary size: {}'.format(tokenizer.vocab_size))

sample_string = 'TensorFlow is cool.'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

# The tokenizer encodes the string by breaking it into subwords
# if the word is not in its dictionary.
for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)


# Create the model
class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.main = keras.Sequential([
            tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
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

# Train the model
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# If the prediction is >= 0.5, it is positive else it is negative.
def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, pad):
    tokenized_sample_pred_text = tokenizer.encode(sample_pred_text)

    if pad:
        tokenized_sample_pred_text = pad_to_size(tokenized_sample_pred_text, 64)

    predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

    return predictions


# predict on a sample text without padding.

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

# predict on a sample text with padding

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

# Stack two or more LSTM layers
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# predict on a sample text without padding.

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

# predict on a sample text with padding

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
