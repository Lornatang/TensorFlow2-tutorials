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

"""To read data efficiently it can be helpful to serialize your data and store
 it in a set of files (100-200MB each) that can each be read linearly.
"""
import tensorflow as tf

import numpy as np
import IPython.display as display

"""## `tf.Example`

### Data types for `tf.Example`

Fundamentally a `tf.Example` is a `{"string": tf.train.Feature}` mapping.

The `tf.train.Feature` message type can accept one of the following three types (See the [`.proto` file]((https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto) for reference). Most other generic types can be coerced into one of these.

1. `tf.train.BytesList` (the following types can be coerced)

- `string`
- `byte`

1. `tf.train.FloatList` (the following types can be coerced)

- `float` (`float32`)
- `double` (`float64`)

1. `tf.train.Int64List` (the following types can be coerced)

- `bool`
- `enum`
- `int32`
- `uint32`
- `int64`
- `uint64`

In order to convert a standard TensorFlow type to a `tf.Example`-compatible `tf.train.Feature`, you can use the following shortcut functions:

Each function takes a scalar input value and returns a `tf.train.Feature` containing one of the three `list` types above.
"""

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
      value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


"""Note: To stay simple, this example only uses scalar inputs. The simplest way to handle non-scalar features is to use `tf.serialize_tensor` to convert tensors to binary-strings. Strings are scalars in tensorflow. Use `tf.parse_tensor` to convert the binary-string back to a tensor.

Below are some examples of how these functions work. Note the varying input types and the standardizes output types. If the input type for a function does not match one of the coercible types stated above, the function will raise an exception (e.g. `_int64_feature(1.0)` will error out, since `1.0` is a float, so should be used with the `_float_feature` function instead).
"""

print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))

"""All proto messages can be serialized to a binary-string using the `.SerializeToString` method."""

feature = _float_feature(np.exp(1))

feature.SerializeToString()

"""### Creating a `tf.Example` message

Suppose you want to create a `tf.Example` message from existing data. In practice, the dataset may come from anywhere, but the procedure of creating the `tf.Example` message from a single observation will be the same.

1. Within each observation, each value needs to be converted to a `tf.train.Feature` containing one of the 3 compatible types, using one of the functions above.

1. We create a map (dictionary) from the feature name string to the encoded feature value produced in #1.

1. The map produced in #2 is converted to a [`Features` message](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto#L85).

In this notebook, we will create a dataset using NumPy.

This dataset will have 4 features.
- a boolean feature, `False` or `True` with equal probability
- an integer feature uniformly randomly chosen from `[0, 5)`
- a string feature generated from a string table by using the integer feature as an index
- a float feature from a standard normal distribution

Consider a sample consisting of 10,000 independently and identically distributed observations from each of the above distributions.
"""

# the number of observations in the dataset
n_observations = int(1e4)

# boolean feature, encoded as False or True
feature0 = np.random.choice([False, True], n_observations)

# integer feature, random from 0 .. 4
feature1 = np.random.randint(0, 5, n_observations)

# string feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)

"""Each of these features can be coerced into a `tf.Example`-compatible type using one of `_bytes_feature`, `_float_feature`, `_int64_feature`. We can then create a `tf.Example` message from these encoded features."""


def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


"""For example, suppose we have a single observation from the dataset, `[False, 4, bytes('goat'), 0.9876]`. We can create and print the `tf.Example` message for this observation using `create_message()`. Each single observation will be written as a `Features` message as per the above. Note that the `tf.Example` [message](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L88) is just a wrapper around the `Features` message."""

# This is an example observation from the dataset.

example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)

"""To decode the message use the `tf.train.Example.FromString` method."""

example_proto = tf.train.Example.FromString(serialized_example)

"""## TFRecords Format Details

A TFRecord file contains a sequence of records. The file can only be read sequentially.

Each record contains a byte-string, for the data-payload, plus the data-length, and  CRC32C (32-bit CRC using the Castagnoli polynomial) hashes for integrity checking.

Each record has the format

  uint64 length
  uint32 masked_crc32_of_length
  byte   data[length]
  uint32 masked_crc32_of_data

The records are concatenated together to produce the file. CRCs are
[described here](https://en.wikipedia.org/wiki/Cyclic_redundancy_check), and
the mask of a CRC is

  masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul

Note: There is no requirement to use `tf.Example` in TFRecord files. `tf.Example` is just a method of serializing dictionaries to byte-strings. Lines of text, encoded  image data, or serialized tensors (using `tf.io.serialize_tensor`, and
`tf.io.parse_tensor` when loading). See the `tf.io` module for more options.

## TFRecord files using `tf.data`

The `tf.data` module also provides tools for reading and writing data in tensorflow.

### Writing a TFRecord file

The easiest way to get the data into a dataset is to use the `from_tensor_slices` method.

Applied to an array, it returns a dataset of scalars.
"""

tf.data.Dataset.from_tensor_slices(feature1)

"""Applies to a tuple of arrays, it returns a dataset of tuples:"""

features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))

# Use `take(1)` to only pull one example from the dataset.
for f0, f1, f2, f3 in features_dataset.take(1):
  print(f0)
  print(f1)
  print(f2)
  print(f3)

"""Use the `tf.data.Dataset.map` method to apply a function to each element of a `Dataset`.

The mapped function must operate in TensorFlow graph mode: It must operate on and return `tf.Tensors`. A non-tensor function, like `create_example`, can be wrapped with `tf.py_function` to make it compatible.

Using `tf.py_function` requires that you specify the shape and type information that is otherwise unavailable:
"""


def tf_serialize_example(f0, f1, f2, f3):
  tf_string = tf.py_function(
      serialize_example,
      (f0, f1, f2, f3),  # pass these args to the above function.
      tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ())  # The result is a scalar


tf_serialize_example(f0, f1, f2, f3)

"""Apply this function to each element in the dataset:"""

serialized_features_dataset = features_dataset.map(tf_serialize_example)


def generator():
  for features in features_dataset:
      yield serialize_example(*features)


serialized_features_dataset = tf.data.Dataset.from_generator(
  generator, output_types=tf.string, output_shapes=())


"""And write them to a TFRecord file:"""

filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

"""### Reading a TFRecord file

We can also read the TFRecord file using the `tf.data.TFRecordDataset` class.

More information on consuming TFRecord files using `tf.data` can be found [here](https://www.tensorflow.org/guide/datasets#consuming_tfrecord_data).

Using `TFRecordDataset`s can be useful for standardizing input data and optimizing performance.
"""

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)

"""At this point the dataset contains serialized `tf.train.Example` messages. When iterated over it returns these as scalar string tensors.

Use the `.take` method to only show the first 10 records.

Note: iterating over a `tf.data.Dataset` only works with eager execution enabled.
"""

for raw_record in raw_dataset.take(10):
  print(repr(raw_record))

"""These tensors can be parsed using the function below.

Note: The `feature_description` is necessary here because datasets use graph-execution, and need this description to build their shape and type signature.
"""

# Create a description of the features.
feature_description = {
  'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
  'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
  'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}


def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


"""Or use `tf.parse example` to parse a whole batch at once.

Apply this finction to each item in the dataset using the `tf.data.Dataset.map` method:
"""

parsed_dataset = raw_dataset.map(_parse_function)

"""Use eager execution to display the observations in the dataset. There are 10,000 observations in this dataset, but we only display the first 10. The data is displayed as a dictionary of features. Each item is a `tf.Tensor`, and the `numpy` element of this tensor displays the value of the feature."""

for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))

"""Here, the `tf.parse_example` function unpacks the `tf.Example` fields into standard tensors.

## TFRecord files in python

The `tf.io` module also contains pure-Python functions for reading and writing TFRecord files.

### Writing a TFRecord file

Now write the 10,000 observations to the file `test.tfrecords`. Each observation is converted to a `tf.Example` message, then written to file. We can then verify that the file `test.tfrecords` has been created.
"""

# Write the `tf.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
      example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
      writer.write(example)


"""### Reading a TFRecord file

These serialized tensores can be easily parsed using `tf.train.Example.ParseFromString`
"""

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)

"""## Walkthrough: Reading/Writing Image Data

This is an example of how to read and write image data using TFRecords. The purpose of this is to show how, end to end, input data (in this case an image) and write the data as a TFRecord file, then read the file back and display the image.

This can be useful if, for example, you want to use several models on the same input dataset. Instead of storing the image data raw, it can be preprocessed into the TFRecords format, and that can be used in all further processing and modelling.

First, let's download [this image](https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg) of a cat in the snow and [this photo](https://upload.wikimedia.org/wikipedia/commons/f/fe/New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg) of the Williamsburg Bridge, NYC under construction.

### Fetch the images
"""

cat_in_snow = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))

display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))

"""### Write the TFRecord file

As we did earlier, we can now encode the features as types compatible with `tf.Example`. In this case, we will not only store the raw image string as a feature, but we will store the height, width, depth, and an arbitrary `label` feature, which is used when we write the file to distinguish between the cat image and the bridge image. We will use `0` for the cat image, and `1` for the bridge image.
"""

image_labels = {
  cat_in_snow: 0,
  williamsburg_bridge: 1,
}

# This is an example, just using the cat image.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

# Create a dictionary with features that may be relevant.


def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')

"""We see that all of the features are now stores in the `tf.Example` message. Now, we functionalize the code above and write the example messages to a file, `images.tfrecords`."""

# Write the raw image files to images.tfrecords.
# First, process the two images into tf.Example messages.
# Then, write to a .tfrecords file.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
      image_string = open(filename, 'rb').read()
      tf_example = image_example(image_string, label)
      writer.write(tf_example.SerializeToString())


"""### Read the TFRecord file

We now have the file `images.tfrecords`. We can now iterate over the records in the file to read back what we wrote. Since, for our use case we will just reproduce the image, the only feature we need is the raw image string. We can extract that using the getters described above, namely `example.features.feature['image_raw'].bytes_list.value[0]`. We also use the labels to determine which record is the cat as opposed to the bridge.
"""

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
  'height': tf.io.FixedLenFeature([], tf.int64),
  'width': tf.io.FixedLenFeature([], tf.int64),
  'depth': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

"""Recover the images from the TFRecord file:"""

for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))
