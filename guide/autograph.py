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

""""""

import tensorflow as tf

import timeit

"""## The `tf.function` decorator

When you annotate a function with `tf.function`, you can still call it like any other function. But it will be compiled into a graph, which means you get the benefits of faster execution, running on GPU or TPU, or exporting to SavedModel.
"""


@tf.function
def simple_nn_layer(x, y):
  return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer(x, y)

"""If we examine the result of the annotation, we can see that it's a special callable that handles all interactions with the TensorFlow runtime."""

"""If your code uses multiple functions, you don't need to annotate them all - any functions called from an annotated function will also run in graph mode."""


def linear_layer(x):
  return 2 * x + 1


@tf.function
def deep_net(x):
  return tf.nn.relu(linear_layer(x))


deep_net(tf.constant((1, 2, 3)))

"""Functions can be faster than eager code, for graphs with many small ops. But for graphs with a few expensive ops (like convolutions), you may not see much speedup."""


conv_layer = tf.keras.layers.Conv2D(100, 3)


@tf.function
def conv_fn(image):
  return conv_layer(image)


image = tf.zeros([1, 200, 200, 100])
# warm up
conv_layer(image)
conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")

lstm_cell = tf.keras.layers.LSTMCell(10)


@tf.function
def lstm_fn(inputs, state):
  return lstm_cell(inputs, state)


inputs = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2
# warm up
lstm_cell(inputs, state)
lstm_fn(inputs, state)
print("eager lstm:", timeit.timeit(lambda: lstm_cell(inputs, state), number=10))
print("function lstm:", timeit.timeit(lambda: lstm_fn(inputs, state), number=10))

"""## Use Python control flow

When using data-dependent control flow inside `tf.function`, you can use Python control flow statements and AutoGraph will convert them into appropriate TensorFlow ops. For example, `if` statements will be converted into `tf.cond()` if they depend on a `Tensor`.

In the example below, `x` is a `Tensor` but the `if` statement works as expected:
"""


@tf.function
def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0
  return x


print('square_if_positive(2) = {}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {}'.format(square_if_positive(tf.constant(-2))))

"""Note: The previous example uses simple conditionals with scalar values. <a href="#batching">Batching</a> is typically used in real-world code.

AutoGraph supports common Python statements like `while`, `for`, `if`, `break`, `continue` and `return`, with support for nesting. That means you can use `Tensor` expressions in the condition of `while` and `if` statements, or iterate over a `Tensor` in a `for` loop.
"""


@tf.function
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s


sum_even(tf.constant([10, 12, 15, 20]))

"""AutoGraph also provides a low-level API for advanced users. For example we can use it to have a look at the generated code."""

print(tf.autograph.to_code(sum_even.python_function))

"""Here's an example of more complicated control flow:"""


@tf.function
def fizzbuzz(n):
  _ = tf.constant('')
  for i in tf.range(n):
    if tf.equal(i % 3, 0):
      tf.print('Fizz')
    elif tf.equal(i % 5, 0):
      tf.print('Buzz')
    else:
      tf.print(i)


fizzbuzz(tf.constant(15))

"""## Keras and AutoGraph

You can use `tf.function` with object methods as well. For example, you can decorate your custom Keras models, typically by annotating the model's `call` function. For more information, see `tf.keras`.
"""


class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      return input_data // 2


model = CustomModel()

model(tf.constant([-2, -4]))

"""## Side effects

Just like in eager mode, you can use operations with side effects, like `tf.assign` or `tf.print` normally inside `tf.function`, and it will insert the necessary control dependencies to ensure they execute in order.
"""

v = tf.Variable(5)


@tf.function
def find_next_odd():
  v.assign(v + 1)
  if tf.equal(v % 2, 0):
    v.assign(v + 1)


find_next_odd()

"""## Example: training a simple model

AutoGraph also allows you to move more computation inside TensorFlow. For example, a training loop is just control flow, so it can actually be brought into TensorFlow.

### Download data
"""


def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y


def mnist_dataset():
  (x, y), _ = tf.keras.datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds


train_dataset = mnist_dataset()

"""### Define the model"""

model = tf.keras.Sequential((
  tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(10)))
model.build()
optimizer = tf.keras.optimizers.Adam()

"""### Define the training loop"""

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss


@tf.function
def train(model, optimizer):
  train_ds = mnist_dataset()
  step = 0
  loss = 0.0
  accuracy = 0.0
  for x, y in train_ds:
    step += 1
    loss = train_one_step(model, optimizer, x, y)
    if tf.equal(step % 10, 0):
      tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
  return step, loss, accuracy


step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())

"""## Batching

In real applications batching is essential for performance. The best code to convert to AutoGraph is code where the control flow is decided at the _batch_ level. If making decisions at the individual _example_ level, try to use batch APIs to maintain performance.

For example, if you have the following code in Python:
"""


def square_if_positive(x):
  return [i ** 2 if i > 0 else i for i in x]


square_if_positive(range(-5, 5))

"""You may be tempted to write it in TensorFlow as such (and this would work!):"""


@tf.function
def square_if_positive_naive(x):
  result = tf.TensorArray(tf.int32, size=x.shape[0])
  for i in tf.range(x.shape[0]):
    if x[i] > 0:
      result = result.write(i, x[i] ** 2)
    else:
      result = result.write(i, x[i])
  return result.stack()


square_if_positive_naive(tf.range(-5, 5))

"""But in this case, it turns out you can write the following:"""


def square_if_positive_vectorized(x):
  return tf.where(x > 0, x ** 2, x)


square_if_positive_vectorized(tf.range(-5, 5))
