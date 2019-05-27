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

import tensorflow as tf

import tensorflow.compat.v1 as tf_compat


class Net(tf.keras.Model):
  """A simple linear model."""

  def __init__(self):
    super(Net, self).__init__()
    self.l1 = tf.keras.layers.Dense(5)

  def call(self, x):
    return self.l1(x)


"""Although it's not the focus of this guide, to be executable the example needs data and an optimization step. The model will train on slices of an in-memory dataset."""


def toy_dataset():
  inputs = tf.range(10.)[:, None]
  labels = inputs * 5. + tf.range(5.)[None, :]
  return tf.data.Dataset.from_tensor_slices(
    dict(x=inputs, y=labels)).repeat(10).batch(2)


def train_step(net, example, optimizer):
  """Trains `net` on `example` using `optimizer`."""
  with tf.GradientTape() as tape:
    output = net(example['x'])
    loss = tf.reduce_mean(tf.abs(output - example['y']))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss


"""The following training loop creates an instance of the model and of an optimizer, then gathers them into a `tf.train.Checkpoint` object. It calls the training step in a loop on each batch of data, and periodically writes checkpoints to disk."""

opt = tf.keras.optimizers.Adam(0.1)
net = Net()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")

for example in toy_dataset():
  loss = train_step(net, example, opt)
  ckpt.step.assign_add(1)
  if int(ckpt.step) % 10 == 0:
    save_path = manager.save()
    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
    print("loss {:1.2f}".format(loss.numpy()))

"""The preceding snippet will randomly initialize the model variables when it first runs. After the first run it will resume training from where it left off:"""

opt = tf.keras.optimizers.Adam(0.1)
net = Net()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")

for example in toy_dataset():
  loss = train_step(net, example, opt)
  ckpt.step.assign_add(1)
  if int(ckpt.step) % 10 == 0:
    save_path = manager.save()
    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
    print("loss {:1.2f}".format(loss.numpy()))

"""The `tf.train.CheckpointManager` object deletes old checkpoints. Above it's configured to keep only the three most recent checkpoints."""

print(manager.checkpoints)  # List the three remaining checkpoints

"""These paths, e.g. `'./tf_ckpts/ckpt-10'`, are not files on disk. Instead they are prefixes for an `index` file and one or more data files which contain the variable values. These prefixes are grouped together in a single `checkpoint` file (`'./tf_ckpts/checkpoint'`) where the `CheckpointManager` saves its state."""

"""<a id="loading_mechanics"/>
## Loading mechanics

TensorFlow matches variables to checkpointed values by traversing a directed graph with named edges, starting from the object being loaded. Edge names typically come from attribute names in objects, for example the `"l1"` in `self.l1 = tf.keras.layers.Dense(5)`. `tf.train.Checkpoint` uses its keyword argument names, as in the `"step"` in `tf.train.Checkpoint(step=...)`.

The dependency graph from the example above looks like this:

![Visualization of the dependency graph for the example training loop](http://tensorflow.org/images/guide/whole_checkpoint.svg)

With the optimizer in red, regular variables in blue, and optimizer slot variables in orange. The other nodes, for example representing the `tf.train.Checkpoint`, are black.

Slot variables are part of the optimizer's state, but are created for a specific variable. For example the `'m'` edges above correspond to momentum, which the Adam optimizer tracks for each variable. Slot variables are only saved in a checkpoint if the variable and the optimizer would both be saved, thus the dashed edges.

Calling `restore()` on a `tf.train.Checkpoint` object queues the requested restorations, restoring variable values as soon as there's a matching path from the `Checkpoint` object. For example we can load just the kernel from the model we defined above by reconstructing one path to it through the network and the layer.
"""

to_restore = tf.Variable(tf.zeros([5]))
print(to_restore.numpy())  # All zeros
fake_layer = tf.train.Checkpoint(bias=to_restore)
fake_net = tf.train.Checkpoint(l1=fake_layer)
new_root = tf.train.Checkpoint(net=fake_net)
status = new_root.restore(tf.train.latest_checkpoint('./tf_ckpts/'))
print(to_restore.numpy())  # We get the restored value now

"""The dependency graph for these new objects is a much smaller subgraph of the larger checkpoint we wrote above. It includes only the bias and a save counter that `tf.train.Checkpoint` uses to number checkpoints.

![Visualization of a subgraph for the bias variable](http://tensorflow.org/images/guide/partial_checkpoint.svg)

`restore()` returns a status object, which has optional assertions. All of the objects we've created in our new `Checkpoint` have been restored, so `status.assert_existing_objects_matched()` passes.
"""

status.assert_existing_objects_matched()

"""There are many objects in the checkpoint which haven't matched, including the layer's kernel and the optimizer's variables. `status.assert_consumed()` only passes if the checkpoint and the program match exactly, and would throw an exception here.

### Delayed restorations

`Layer` objects in TensorFlow may delay the creation of variables to their first call, when input shapes are available. For example the shape of a `Dense` layer's kernel depends on both the layer's input and output shapes, and so the output shape required as a constructor argument is not enough information to create the variable on its own. Since calling a `Layer` also reads the variable's value, a restore must happen between the variable's creation and its first use.

To support this idiom, `tf.train.Checkpoint` queues restores which don't yet have a matching variable.
"""

delayed_restore = tf.Variable(tf.zeros([1, 5]))
print(delayed_restore.numpy())  # Not restored; still zeros
fake_layer.kernel = delayed_restore
print(delayed_restore.numpy())  # Restored

"""### Manually inspecting checkpoints

`tf.train.list_variables` lists the checkpoint keys and shapes of variables in a checkpoint. Checkpoint keys are paths in the graph displayed above.
"""

tf.train.list_variables(tf.train.latest_checkpoint('./tf_ckpts/'))

"""### List and dictionary tracking

As with direct attribute assignments like `self.l1 = tf.keras.layers.Dense(5)`, assigning lists and dictionaries to attributes will track their contents.
"""

save = tf.train.Checkpoint()
save.listed = [tf.Variable(1.)]
save.listed.append(tf.Variable(2.))
save.mapped = {'one': save.listed[0]}
save.mapped['two'] = save.listed[1]
save_path = save.save('./tf_list_example')

restore = tf.train.Checkpoint()
v2 = tf.Variable(0.)
assert 0. == v2.numpy()  # Not restored yet
restore.mapped = {'two': v2}
restore.restore(save_path)
assert 2. == v2.numpy()

"""You may notice wrapper objects for lists and dictionaries. These wrappers are checkpointable versions of the underlying data-structures. Just like the attribute based loading, these wrappers restore a variable's value as soon as it's added to the container."""

restore.listed = []
print(restore.listed)  # ListWrapper([])
v1 = tf.Variable(0.)
restore.listed.append(v1)  # Restores v1, from restore() in the previous cell
assert 1. == v1.numpy()

"""The same tracking is automatically applied to subclasses of `tf.keras.Model`, and may be used for example to track lists of layers.

## Saving object-based checkpoints with Estimator

See the [guide to Estimator](https://www.tensorflow.org/guide/estimators).

Estimators by default save checkpoints with variable names rather than the object graph described in the previous sections. `tf.train.Checkpoint` will accept name-based checkpoints, but variable names may change when moving parts of a model outside of the Estimator's `model_fn`. Saving object-based checkpoints makes it easier to train a model inside an Estimator and then use it outside of one.
"""


def model_fn(features, labels, mode):
  net = Net()
  opt = tf.keras.optimizers.Adam(0.1)
  ckpt = tf.train.Checkpoint(step=tf_compat.train.get_global_step(),
                             optimizer=opt, net=net)
  with tf.GradientTape() as tape:
    output = net(features['x'])
    loss = tf.reduce_mean(tf.abs(output - features['y']))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  return tf.estimator.EstimatorSpec(
    mode,
    loss=loss,
    train_op=tf.group(opt.apply_gradients(zip(gradients, variables)),
                      ckpt.step.assign_add(1)),
    # Tell the Estimator to save "ckpt" in an object-based format.
    scaffold=tf_compat.train.Scaffold(saver=ckpt))


tf.keras.backend.clear_session()
est = tf.estimator.Estimator(model_fn, './tf_estimator_example/')
est.train(toy_dataset, steps=10)

"""`tf.train.Checkpoint` can then load the Estimator's checkpoints from its `model_dir`."""

opt = tf.keras.optimizers.Adam(0.1)
net = Net()
ckpt = tf.train.Checkpoint(
  step=tf.Variable(1, dtype=tf.int64), optimizer=opt, net=net)
ckpt.restore(tf.train.latest_checkpoint('./tf_estimator_example/'))
ckpt.step.numpy()  # From est.train(..., steps=10)

"""## Summary

TensorFlow objects provide an easy automatic mechanism for saving and restoring the values of variables they use.
"""
