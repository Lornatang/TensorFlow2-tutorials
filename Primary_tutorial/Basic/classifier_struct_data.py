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
"""This tutorial demonstrates how to classify structured data (e.g. tabular data in a CSV).
   We will use Keras to define the model, and feature columns as a bridge to map
   from columns in a CSV to features used to train the model.
   This tutorial contains complete code to:

   - Load a CSV file using Pandas.
   - Build an input pipeline to batch and shuffle the rows using tf.data.
   - Map from columns in the CSV to features used to train the model using feature columns.
   - Build, train, and evaluate a model using Keras.
"""

# import TensorFlow and other libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.python.keras import layers

# Use Pandas to create a dataFrame.
# Pandas is a Python library with many helpful utilities for loading and working with structured data.
# We will use Pandas to download the dataset from a URL, and load it into a dataFrame.
URL = './heart.csv'
dataFrame = pd.read_csv(URL)
dataFrame.head()

# Split the dataFrame into train, validation, and test.
# The dataset we downloaded was a single CSV file.
# We will split this into train, validation, and test sets.

train, test = train_test_split(dataFrame, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Create an input pipeline using tf.data
# Next, we will wrap the dataframes with tf.data.
# This will enable us to use feature columns as a bridge to map
# from the columns in the Pandas dataFrame to features used to train the model.
# If we were working with a very large CSV file (so large that it does not fit into memory),
# we would use tf.data to read it from disk directly. That is not covered in this tutorial.

# A utility method to create a tf.data dataset from a Pandas DataFrame


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Understand the input pipeline
# Now that we have created the input pipeline, let's call it to see the format of the data it returns.
# We have used a small batch size to keep the output readable.

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch)

# We can see that the dataset returns a dictionary of column names (from the dataframe) that map to column values
# from rows in the dataframe.

# Demonstrate several types of feature column
# TensorFlow provides many types of feature columns.
# In this section, we will create several types of feature columns, and demonstrate
# how they transform a column from the dataframe.

# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]


# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

# Numeric columns
# The output of a feature column becomes the input to the model (using the demo function defined above,
# we will be able to see exactly how each column from the dataframe is transformed).
# A numeric column is the simplest type of column. It is used to represent real valued features.
# When using this column, your model will receive the column value from the dataframe unchanged.


age = feature_column.numeric_column("age")
demo(age)

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

# Categorical columns
# In this dataset, thal is represented as a string (e.g. 'fixed', 'normal', or 'reversible').
# We cannot feed strings directly to a model. Instead, we must first map them to numeric values.
# The categorical vocabulary columns provide a way to represent strings as a one-hot vector (much
# like you have seen above with age buckets).
# The vocabulary can be passed as a list using categorical_column_with_vocabulary_list,
# or loaded from a file using categorical_column_with_vocabulary_file.

thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# we previously created
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

# Crossed feature columns
# Combining features into a single feature, better known as feature crosses,
# enables a model to learn separate weights for each combination of features.
# Here, we will create a new feature that is the cross of age and thal.
# Note that crossed_column does not build the full table of all possible
# combinations (which could be very large). Instead, it is backed by a hashed_column,
# so you can choose how large the table is.

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Create, compile, and train the model
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
