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

"""
In TensorFlow 2.0, eager execution is turned on by default.
The user interface is intuitive and flexible
(running one-off operations is much easier and faster),
but this can come at the expense of performance and deployability.
"""

# Import TensorFlow and other library
import tensorflow as tf

import contextlib

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print(f'Caught expected exception \n  {error_class}: {e}')
  except Exception as e:
    print(f'Got unexpected exception \n  {type(e)}: {e}')
  else:
    raise Exception(f'Expected {error_class} to be raised but no error was raised!')


# A function is like an op
@tf.function
def add(a, b):
  return a + b


print(add(tf.ones([2, 2]), tf.ones([2, 2])))  # [[2., 2.], [2., 2.]]
print()


# Functions have gradients
@tf.function
def add(a, b):
  return a + b


v = tf.Variable(1.0)

with tf.GradientTape() as tape:
  result = add(v, 1.0)
print(tape.gradient(result, v))
print()


# You can use functions inside functions
@tf.function
def dense_layer(x, w, b):
  return add(tf.matmul(x, w), b)


print(dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2])))
print()


# Tracing and polymorphism
# Functions are polymorphic
@tf.function
def double(a):
  print("Tracing with", a)
  print()
  return a + a


print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()

print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("Using a concrete trace with incompatible types will throw an error")
with assert_raises(tf.errors.InvalidArgumentError):
  double_strings(tf.constant(1))


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
  print("Tracing with", x)
  return tf.where(tf.equal(x % 2, 0), x // 2, 3 * x + 1)


print(next_collatz(tf.constant([1, 2])))
# We specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
  next_collatz(tf.constant([[1, 2], [3, 4]]))


def train_one_step():
  pass


@tf.function
def train(num_steps):
  print("Tracing with num_steps = {}".format(num_steps))
  for _ in tf.range(num_steps):
    train_one_step()


train(num_steps=10)
train(num_steps=20)

train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))


# Side effects in tf.function
@tf.function
def f(x):
  print("Traced with", x)
  tf.print("Executed with", x)


f(1)
f(1)
f(2)

external_list = []


def side_effect(x):
  print('Python side effect')
  external_list.append(x)


@tf.function
def f(x):
  tf.py_function(side_effect, inp=[x], Tout=[])


f(1)
f(1)
f(1)
assert len(external_list) == 3
# .numpy() call required because py_function casts 1 to tf.constant(1)
assert external_list[0].numpy() == 1

# Beware of Python state
external_var = tf.Variable(0)


@tf.function
def buggy_consume_next(iterator):
  external_var.assign_add(next(iterator))
  tf.print("Value of external_var:", external_var)


iterator = iter([0, 1, 2, 3])
buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)


def measure_graph_size(f, *args):
  g = f.get_concrete_function(*args).graph
  print("{}({}) contains {} nodes in its graph".format(
    f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))


@tf.function
def train(dataset):
  loss = tf.constant(0)
  for x, y in dataset:
    loss += tf.abs(y - x)  # Some dummy computation.
  return loss


small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
  lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
  lambda: big_data, (tf.int32, tf.int32)))

# Automatic Control Dependencies
# Automatic control dependencies

a = tf.Variable(1.0)
b = tf.Variable(2.0)


@tf.function
def f(x, y):
  a.assign(y * b)
  b.assign_add(x * a)
  return a + b


f(1.0, 2.0)  # 10.0


# Variables
@tf.function
def f(x):
  v = tf.Variable(1.0)
  v.assign_add(x)
  return v


with assert_raises(ValueError):
  f(1.0)

# Non-ambiguous code is ok though

v = tf.Variable(1.0)


@tf.function
def f(x):
  return v.assign_add(x)


print(f(1.0))  # 2.0
print(f(2.0))  # 4.0


# You can also create variables inside a tf.function as long as we can prove
# that those variables are created only the first time the function is executed.

class C:
  pass


obj = C()
obj.v = None


@tf.function
def g(x):
  if obj.v is None:
    obj.v = tf.Variable(1.0)
  return obj.v.assign_add(x)


print(g(1.0))  # 2.0
print(g(2.0))  # 4.0

# Variable initializers can depend on function arguments and on values of other
# variables. We can figure out the right initialization order using the same
# method we use to generate control dependencies.

state = []


@tf.function
def fn(x):
  if not state:
    state.append(tf.Variable(2.0 * x))
    state.append(tf.Variable(state[0] * 3.0))
  return state[0] * x * state[1]


print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))


# Using AutoGraph
# Simple loop

@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x


f(tf.random.uniform([5]))


# If you're curious you can inspect the code autograph generates.
# It feels like reading assembly language, though.

def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x


print(tf.autograph.to_code(f))


# AutoGraph: Conditionals
def test_tf_cond(f, *args):
  g = f.get_concrete_function(*args).graph
  if any(node.name == 'cond' for node in g.as_graph_def().node):
    print("{}({}) uses tf.cond.".format(
      f.__name__, ', '.join(map(str, args))))
  else:
    print("{}({}) executes normally.".format(
      f.__name__, ', '.join(map(str, args))))


@tf.function
def hyperparam_cond(x, training=True):
  if training:
    x = tf.nn.dropout(x, rate=0.5)
  return x


@tf.function
def maybe_tensor_cond(x):
  if x < 0:
    x = -x
  return x


test_tf_cond(hyperparam_cond, tf.ones([1], dtype=tf.float32))
test_tf_cond(maybe_tensor_cond, tf.constant(-1))
test_tf_cond(maybe_tensor_cond, -1)


@tf.function
def f():
  x = tf.constant(0)
  if tf.constant(True):
    x = x + 1
    print("Tracing `then` branch")
  else:
    x = x - 1
    print("Tracing `else` branch")
  return x


f()


# AutoGraph and loops
def test_dynamically_unrolled(f, *args):
  g = f.get_concrete_function(*args).graph
  if any(node.name == 'while' for node in g.as_graph_def().node):
    print("{}({}) uses tf.while_loop.".format(
      f.__name__, ', '.join(map(str, args))))
  elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
    print("{}({}) uses tf.data.Dataset.reduce.".format(
      f.__name__, ', '.join(map(str, args))))
  else:
    print("{}({}) gets unrolled.".format(
      f.__name__, ', '.join(map(str, args))))


@tf.function
def for_in_range():
  x = 0
  for i in range(5):
    x += i
  return x


@tf.function
def for_in_tfrange():
  x = tf.constant(0, dtype=tf.int32)
  for i in tf.range(5):
    x += i
  return x


@tf.function
def for_in_tfdataset():
  x = tf.constant(0, dtype=tf.int64)
  for i in tf.data.Dataset.range(5):
    x += i
  return x


test_dynamically_unrolled(for_in_range)
test_dynamically_unrolled(for_in_tfrange)
test_dynamically_unrolled(for_in_tfdataset)


@tf.function
def while_py_cond():
  x = 5
  while x > 0:
    x -= 1
  return x


@tf.function
def while_tf_cond():
  x = tf.constant(5)
  while x > 0:
    x -= 1
  return x


test_dynamically_unrolled(while_py_cond)
test_dynamically_unrolled(while_tf_cond)


@tf.function
def buggy_while_py_true_tf_break(x):
  while True:
    if tf.equal(x, 0):
      break
    x -= 1
  return x


@tf.function
def while_tf_true_tf_break(x):
  while tf.constant(True):
    if tf.equal(x, 0):
      break
    x -= 1
  return x


with assert_raises(TypeError):
  test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)
test_dynamically_unrolled(while_tf_true_tf_break, 5)


@tf.function
def buggy_py_for_tf_break():
  x = 0
  for i in range(5):
    if tf.equal(i, 3):
      break
    x += i
  return x


@tf.function
def tf_for_tf_break():
  x = 0
  for i in tf.range(5):
    if tf.equal(i, 3):
      break
    x += i
  return x


with assert_raises(TypeError):
  test_dynamically_unrolled(buggy_py_for_tf_break)
test_dynamically_unrolled(tf_for_tf_break)

batch_size = 2
seq_len = 3
feature_size = 4


def rnn_step(inp, state):
  return inp + state


@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
  # [batch, time, features] -> [time, batch, features]
  input_data = tf.transpose(input_data, [1, 0, 2])
  max_seq_len = input_data.shape[0]

  states = tf.TensorArray(tf.float32, size=max_seq_len)
  state = initial_state
  for i in tf.range(max_seq_len):
    state = rnn_step(input_data[i], state)
    states = states.write(i, state)
  return tf.transpose(states.stack(), [1, 0, 2])


dynamic_rnn(rnn_step,
            tf.random.uniform([batch_size, seq_len, feature_size]),
            tf.zeros([batch_size, feature_size]))


@tf.function
def buggy_loop_var_uninitialized():
  for i in tf.range(3):
    x = i
  return x


@tf.function
def f():
  x = tf.constant(0)
  for i in tf.range(3):
    x = i
  return x


with assert_raises(ValueError):
  buggy_loop_var_uninitialized()
f()


@tf.function
def buggy_loop_type_changes():
  x = tf.constant(0, dtype=tf.float32)
  for i in tf.range(3):  # Yields tensors of type tf.int32...
    x = i
  return x


with assert_raises(tf.errors.InvalidArgumentError):
  buggy_loop_type_changes()


@tf.function
def buggy_concat():
  x = tf.ones([0, 10])
  for _ in tf.range(5):
    x = tf.concat([x, tf.ones([1, 10])], axis=0)
  return x


with assert_raises(ValueError):
  buggy_concat()


@tf.function
def concat_with_padding():
  x = tf.zeros([5, 10])
  for i in tf.range(5):
    x = tf.concat([x[:i], tf.ones([1, 10]), tf.zeros([4 - i, 10])], axis=0)
    x.set_shape([5, 10])
  return x


concat_with_padding()
