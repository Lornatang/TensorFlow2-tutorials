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

"""In the previous tutorial we covered the TensorFlow APIs for automatic differentiation,
   a basic building block for machine learning.
   In this tutorial we will use the TensorFlow primitives introduced in the
   prior tutorials to do some simple machine learning.
"""

# Import TensorFlow and other library
import tensorflow as tf

import matplotlib.pyplot as plt

# Using python state
x = tf.zeros([10, 10])

# This is equivalent to x = x + 2, which does not mutate the original
# value of x
x += 2

print(x)

v = tf.Variable(1.0)
print(v.numpy())
assert v.numpy() == 1.0

# Re-assign the value
v.assign(3.0)
print(v.numpy())
assert v.numpy() == 3.0

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
print(v.numpy())
assert v.numpy() == 9.0


# Define the model
class Model(object):
  def __init__(self):
    # Initialize variable to (5.0, 0.0)
    # In practice, these should be initialized to random values.
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.W * x + self.b


model = Model()

print(model(3.0).numpy())
assert model(3.0).numpy() == 15.0


# Define a loss function
def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y))


# Obtain training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise


plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: '),
print(loss(model(inputs), outputs).numpy())


# Define training loop
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)


# Finally, let's repeatedly run through the training data and see how W and b evolve.
model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(20)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(inputs), outputs)

  train(model, inputs, outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

"""
Epoch  0: W=5.00 b=0.00, loss=9.17734
Epoch  1: W=4.60 b=0.41, loss=6.22211
Epoch  2: W=4.28 b=0.74, loss=4.33984
Epoch  3: W=4.02 b=1.00, loss=3.14097
Epoch  4: W=3.82 b=1.21, loss=2.37736
Epoch  5: W=3.66 b=1.37, loss=1.89098
Epoch  6: W=3.53 b=1.50, loss=1.58119
Epoch  7: W=3.42 b=1.61, loss=1.38386
Epoch  8: W=3.34 b=1.69, loss=1.25817
Epoch  9: W=3.27 b=1.76, loss=1.17811
Epoch 10: W=3.22 b=1.81, loss=1.12711
Epoch 11: W=3.18 b=1.86, loss=1.09463
Epoch 12: W=3.14 b=1.89, loss=1.07393
Epoch 13: W=3.11 b=1.92, loss=1.06075
Epoch 14: W=3.09 b=1.94, loss=1.05236
Epoch 15: W=3.08 b=1.95, loss=1.04701
Epoch 16: W=3.06 b=1.97, loss=1.04360
Epoch 17: W=3.05 b=1.98, loss=1.04143
Epoch 18: W=3.04 b=1.99, loss=1.04005
Epoch 19: W=3.03 b=1.99, loss=1.03917
"""

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()

