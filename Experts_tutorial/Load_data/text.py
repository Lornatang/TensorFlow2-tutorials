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

"""This tutorial provides an example of how to use tf.data.TextLineDataset to
 load examples from text files.
"""

# Setup
import tensorflow as tf
import tensorflow_datasets as tfds

import os

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
  text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = os.path.dirname(text_dir)
print(f"Dataset saved to '{parent_dir}'.")


# Load text into datasets
def labeler(example, index):
  return example, tf.cast(index, tf.int64)


labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  lines_datasets = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  labeled_datasets = lines_datasets.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_datasets)

# Combine these labeled datasets into a single dataset, and shuffle it.
BUFFER_SIZE = 50000
BATCH_SIZE = 32
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
  BUFFER_SIZE, reshuffle_each_iteration=False)

# Encode text lines as numbers
tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
print(vocab_size)

# Encode examples
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

# You can try this on a single line to see what the output looks like.
example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)

encoded_example = encoder.encode(example_text)
print(encoded_example)


# Now run the encoder on the dataset by wrapping it in tf.py_function and passing that to the dataset's map method.
def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label


def encode_map_fn(text, label):
  return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


all_encoded_data = all_labeled_data.map(encode_map_fn)

# Split the dataset into text and train batches
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

sample_text, sample_labels = next(iter(test_data))
print(sample_text[0], sample_labels[0])

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
