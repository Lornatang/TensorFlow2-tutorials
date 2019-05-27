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

"""This tutorial provides a simple example of how to load an image data set using tf.data."""

# Import TensorFlow and other libraries
import tensorflow as tf
from tensorflow.python import keras
import pathlib

import random
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Load data
data_root_orig = keras.utils.get_file(
  origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
  fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)

# Check images
attributions = (data_root / "LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

# Identify the label for each image
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

# Assign an index to each label.
label_to_index = dict((name, index) for index, name in enumerate(label_names))

# Create a list of each file and its label index
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                  for path in all_image_paths]

# Loads and formats images
img_path = all_image_paths[0]
img_raw = tf.io.read_file(img_path)
img_tensor = tf.image.decode_image(img_raw)

# resize img size
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final / 255.


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def caption_image(image_path):
  image_rel = pathlib.Path(image_path).relative_to(data_root)
  return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)

# Training
BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)).batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

# Pipe the dataset to a model

mobile_net = keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)

mobile_net.trainable = False


def change_range(image, label):
  return 2 * image - 1, label


keras_ds = ds.map(change_range)

# Pass it a batch of images to see:
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))
])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"])

model.summary()

# Normally you would specify the real number of steps per epoch, but for demonstration purposes only run 3 steps.
steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()

model.fit(ds, epochs=1, steps_per_epoch=3)

# ==========================Performance==========================================

# To investigate, first build a simple function to check the performance of
# our datasets.
default_timeit_steps = 2 * steps_per_epoch + 1


def timeit(ds, steps=default_timeit_steps):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(steps + 1))
  next(it)

  start = time.time()
  for i, (_, _) in enumerate(it):
      if i % 10 == 0:
          print('.', end='')
  print()
  end = time.time()

  duration = end - start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))
  print("Total time: {}s".format(end - overall_start))


# The performance of the current dataset is.
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print()
print(f'Basic pipe:')
print()
timeit(ds)
"""
........................ 
231.0 batches: 16.38393759727478 s 
451.17359 Images/s 
Total time: 24.00374174118042s
"""

# Cache
# Here the images are cached, after being pre-precessed (decoded and resized).
ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print()
print(f'Add memory cache pipe (one train.):')
print()
timeit(ds)
""""
........................ 
231.0 batches: 0.5540146827697754 s 
13342.60667 Images/s 
Total time: 8.019374370574951s
"""

# One disadvantage to using an in memory cache is that the cache must be
# rebuilt on each run, giving the same startup delay each time the dataset is started.
print()
print(f'Add memory cache pipe (two train.):')
print()
timeit(ds)
"""
........................ 
231.0 batches: 0.5502834320068359 s 
13433.07752 Images/s 
Total time: 8.023614883422852s
"""

# If the data doesn't fit in memory, use a cache file:
ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)

print()
print(f'Add file cache pipe (one train.):')
print()
timeit(ds)
"""
........................ 
231.0 batches: 4.62050461769104 s 
1599.82526 Images/s 
Total time: 17.26514196395874s
"""

# The cache file also has the advantage that it can be used to quickly restart
# the dataset without rebuilding the cache.
# Note how much faster it is the second time:

print()
print(f'Add file cache pipe (two train.):')
print()
timeit(ds)
"""
........................ 
231.0 batches: 1.8834733963012695 s 
3924.66388 Images/s 
Total time: 2.96774959564209s
"""

# ==========================Performance==========================================

# ==========================TFRecord File========================================

# First, build a TFRecord file from the raw image data.
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)

# Next build a dataset that reads from the TFRecord file and decodes/reformats
# the images using the preprocess_image function we defined earlier.
image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

# Zip that with the labels dataset we defined earlier,
# to get the expected (image,label) pairs.
ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print()
print(f'Add tfrecord file pipe (basic tensor):')
print()
timeit(ds)
"""
........................ 
231.0 batches: 16.974323272705078 s 
435.48128 Images/s 
Total time: 24.889507055282593s
"""
# This is slower than the cache version because we have not cached the preprocessing.

# Serialized Tensors
# To save some preprocessing to the TFRecord file,
# first make a dataset of the processed images, as before.
paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)

ds = image_ds.map(tf.io.serialize_tensor)

tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(ds)

ds = tf.data.TFRecordDataset('images.tfrec')


def parse(x):
  result = tf.io.parse_tensor(x, out_type=tf.float32)
  result = tf.reshape(result, [192, 192, 3])
  return result


ds = ds.map(parse, num_parallel_calls=AUTOTUNE)

# Now, add the labels and apply the same standard operations as before:
ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print()
print(f'Add tfrecord pipe (serialized tensor):')
print()
timeit(ds)
"""
........................ 
231.0 batches: 1.7946977615356445 s 
4118.79936 Images/s 
Total time: 2.6123740673065186s
"""

# ==========================TFRecord File========================================
