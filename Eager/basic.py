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

"""Tensors and Operations"""

# Import TensorFlow
import tensorflow as tf

# Import other library
import numpy as np
import time
import tempfile

# These operations automatically convert native Python types, for example.
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

"""
tf.Tensor(3, shape=(), dtype=int32) 
tf.Tensor([4 6], shape=(2,), dtype=int32) 
tf.Tensor(25, shape=(), dtype=int32) 
tf.Tensor(6, shape=(), dtype=int32) 
tf.Tensor(13, shape=(), dtype=int32)
"""

# Each tf.Tensor has a shape and a datatype.
x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)

"""
tf.Tensor([[2 3]], shape=(1, 2), dtype=int32) 
(1, 2) 
<dtype: 'int32'>
"""

# Converting between a TensorFlow tf.Tensors and a NumPy ndarray is easy.
# - TensorFlow operations automatically convert NumPy ndarrays to Tensors.
# - NumPy operations automatically convert Tensors to NumPy ndarrays.
ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

"""
TensorFlow operations convert numpy arrays to Tensors automatically 
tf.Tensor( 
  [[42. 42. 42.] 
   [42. 42. 42.]
   [42. 42. 42.]], shape=(3, 3), dtype=float64) 
And NumPy operations convert Tensors to numpy arrays automatically 
  [[43. 43. 43.] 
   [43. 43. 43.] 
   [43. 43. 43.]] 
The .numpy() method explicitly converts a Tensor to a numpy array 
  [[42. 42. 42.] 
   [42. 42. 42.] 
   [42. 42. 42.]]
"""

# GPU acceleration
x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

"""
Is there a GPU available: 
False 
Is the Tensor on GPU #0: 
False
"""


# Explicit Device Placement
def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  print("On GPU:")
  with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

"""
On CPU: 10 loops: 88.60ms
"""

# Create a source Dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

# Apply transformations
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

# Iterate
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)

"""
Elements of ds_tensors: 
tf.Tensor([1 9], shape=(2,), dtype=int32) 
tf.Tensor([ 4 25], shape=(2,), dtype=int32) 
tf.Tensor([16 36], shape=(2,), dtype=int32) 
Elements in ds_file: 
tf.Tensor([b'Line 1' b'Line 2'], shape=(2,), dtype=string) 
tf.Tensor([b'Line 3' b' '], shape=(2,), dtype=string)
"""
