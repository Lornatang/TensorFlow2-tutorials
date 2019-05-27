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

"""This tutorial provides an example of how to load CSV data from a file into a tf.data.Dataset."""

import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

"""As you can see, the columns in the CSV are labeled. We need the list later on, so let's read it out of the file."""

# CSV columns in the input file.
with open(train_file_path, 'r') as f:
  names_row = f.readline()

CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)

"""The dataset constructor will pick these labels up automatically.

If the file you are working with does not contain the column names in the first line, pass them in a list of strings to  the `column_names` argument in the `make_csv_dataset` function.

```python

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

dataset = tf.data.experimental.make_csv_dataset(
     ...,
     column_names=CSV_COLUMNS,
     ...)

```

This example is going to use all the available columns. If you need to omit some columns from the dataset, create a list of just the columns you plan to use, and pass it into the (optional) `select_columns` argument of the constructor.


```python

drop_columns = ['fare', 'embark_town']
columns_to_use = [col for col in CSV_COLUMNS if col not in drop_columns]

dataset = tf.data.experimental.make_csv_dataset(
  ...,
  select_columns = columns_to_use, 
  ...)

```

We also have to identify which column will serve as the labels for each example, and what those labels are.
"""

LABELS = [0, 1]
LABEL_COLUMN = 'survived'

FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]

"""Now that these constructor argument values are in place,  read the CSV data from the file and create a dataset. 

(For the full documentation, see `tf.data.experimental.make_csv_dataset`)
"""


def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size=12,  # Artificially small to make examples easier to show.
    label_name=LABEL_COLUMN,
    na_value="?",
    num_epochs=1,
    ignore_errors=True)
  return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

"""Each item in the dataset is a batch, represented as a tuple of (*many examples*, *many labels*). The data from the examples is organized in column-based tensors (rather than row-based tensors), each with as many elements as the batch size (12 in this case).

It might help to see this yourself.
"""

examples, labels = next(iter(raw_train_data))  # Just the first batch.
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

"""## Data preprocessing

### Categorical data

Some of the columns in the CSV data are categorical columns. That is, the content should be one of a limited set of options.

In the CSV, these options are represented as text. This text needs to be converted to numbers before the model can be trained. To facilitate that, we need to create a list of categorical columns, along with a list of the options available in each column.
"""

CATEGORIES = {
  'sex': ['male', 'female'],
  'class': ['First', 'Second', 'Third'],
  'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
  'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
  'alone': ['y', 'n']
}

"""Write a function that takes a tensor of categorical values, matches it to a list of value names, and then performs a one-hot encoding."""


def process_categorical_data(data, categories):
  """Returns a one-hot encoded tensor representing categorical values."""

  # Remove leading ' '.
  data = tf.strings.regex_replace(data, '^ ', '')
  # Remove trailing '.'.
  data = tf.strings.regex_replace(data, r'\.$', '')

  # ONE HOT ENCODE
  # Reshape data from 1d (a list) to a 2d (a list of one-element lists)
  data = tf.reshape(data, [-1, 1])
  # For each element, create a new list of boolean values the length of categories,
  # where the truth value is element == category label
  data = tf.equal(categories, data)
  # Cast booleans to floats.
  data = tf.cast(data, tf.float32)

  # The entire encoding can fit on one line:
  # data = tf.cast(tf.equal(categories, tf.reshape(data, [-1, 1])), tf.float32)
  return data


"""To help you visualize this, we'll take a single category-column tensor from the first batch, preprocess it, and show the before and after state."""

class_tensor = examples['class']

class_categories = CATEGORIES['class']

processed_class = process_categorical_data(class_tensor, class_categories)

"""Notice the relationship between the lengths of the two inputs and the shape of the output."""

print("Size of batch: ", len(class_tensor.numpy()))
print("Number of category labels: ", len(class_categories))
print("Shape of one-hot encoded tensor: ", processed_class.shape)

"""### Continuous data

Continuous data needs to be normalized, so that the values fall between 0 and 1. To do that, write a function that multiplies each value by 1 over twice the mean of the column values.

The function should also reshape the data into a two dimensional tensor.
"""


def process_continuous_data(data, mean):
  # Normalize data
  data = tf.cast(data, tf.float32) * 1 / (2 * mean)
  return tf.reshape(data, [-1, 1])


"""To do this calculation, you need the column means. You would obviously need to compute these in real life, but for this example we'll just provide them."""

MEANS = {
  'age': 29.631308,
  'n_siblings_spouses': 0.545455,
  'parch': 0.379585,
  'fare': 34.385399
}

"""Again, to see what this function is actually doing, we'll take a single tensor of continuous data and show it before and after processing."""

age_tensor = examples['age']

process_continuous_data(age_tensor, MEANS['age'])

"""### Preprocess the data

Now assemble these preprocessing tasks into a single function that can be mapped to each batch in the dataset.
"""


def preprocess(features, labels):
  # Process categorial features.
  for feature in CATEGORIES.keys():
    features[feature] = process_categorical_data(features[feature],
                                                 CATEGORIES[feature])

  # Process continuous features.
  for feature in MEANS.keys():
    features[feature] = process_continuous_data(features[feature],
                                                MEANS[feature])

  # Assemble features into a single tensor.
  features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)

  return features, labels


"""Now apply that function with `tf.Dataset.map`, and shuffle the dataset to avoid overfitting."""

train_data = raw_train_data.map(preprocess).shuffle(500)
test_data = raw_test_data.map(preprocess)

"""And let's see what a single example looks like."""

examples, labels = next(iter(train_data))

"""The examples are in a  two dimensional arrays of 12 items each (the batch size). Each item represents a single row in the original CSV file. The labels are a 1d tensor of 12 values.

## Build the model

This example uses the [Keras Functional API](https://www.tensorflow.org/alpha/guide/keras/functional) wrapped in a `get_model` constructor to build up a simple model.
"""


def get_model(input_dim, hidden_units=None):
  """Create a Keras model with layers.

  Args:
    input_dim: (int) The shape of an item in a batch.
    hidden_units: [int] the layer sizes of the DNN (input layer first)

  Returns:
    A Keras model.
  """

  if hidden_units is None:
    hidden_units = [100]
  inputs = tf.keras.Input(shape=(input_dim,))
  x = inputs

  for units in hidden_units:
    x = tf.keras.layers.Dense(units, activation='relu')(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs, outputs)

  return model


"""The `get_model` constructor needs to know the input shape of your data (not including the batch size)."""

input_shape, output_shape = train_data.output_shapes

input_dimension = input_shape.dims[1]  # [0] is the batch size

"""## Train, evaluate, and predict

Now the model can be instantiated and trained.
"""

model = get_model(input_dimension)
model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

model.fit(train_data, epochs=20)

"""Once the model is trained, we can check its accuracy on the `test_data` set."""

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

"""Use `tf.keras.Model.predict` to infer labels on a batch or a dataset of batches."""

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))
