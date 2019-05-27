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
from tensorflow.python import keras
from tensorflow.python.keras import layers

import numpy as np

tf.keras.backend.clear_session()  # For easy reset of notebook state.

"""## Part I: Saving Sequential models or Functional models

Let's consider the following model:
"""

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

"""Optionally, let's train this model, just so it has weight values to save, as well as an an optimizer state.
Of course, you can save models you've never trained, too, but obviously that's less interesting.
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

# Save predictions for future checks
predictions = model.predict(x_test)

"""### Whole-model saving

You can save a model built with the Functional API into a single file. You can later recreate the same model from this file, even if you no longer have access to the code that created the model.

This file includes:

- The model's architecture
- The model's weight values (which were learned during training)
- The model's training config (what you passed to `compile`), if any
- The optimizer and its state, if any (this enables you to restart training where you left off)
"""

# Save the model
model.save('path_to_my_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')



# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.

"""### Export to SavedModel

You can also export a whole model to the TensorFlow `SavedModel` format. `SavedModel` is a standalone serialization format for Tensorflow objects, supported by TensorFlow serving as well as TensorFlow implementations other than Python.
"""

# Export the model to a SavedModel
keras.experimental.export_saved_model(model, 'path_to_saved_model')

# Recreate the exact same model
new_model = keras.experimental.load_from_saved_model('path_to_saved_model')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.

"""The `SavedModel` files that were created contain:

- A TensorFlow checkpoint containing the model weights.
- A `SavedModel` proto containing the underlying Tensorflow graph. Separate
     graphs are saved for prediction (serving), train, and evaluation. If
     the model wasn't compiled before, then only the inference graph
     gets exported.
- The model's architecture config, if available.

### Architecture-only saving

Sometimes, you are only interested in the architecture of the model, and you don't need to save the weight values or the optimizer. In this case, you can retrieve the "config" of the model via the `get_config()` method. The config is a Python dict that enables you to recreate the same model -- initialized from scratch, without any of the information learned previously during training.
"""

config = model.get_config()
reinitialized_model = keras.Model.from_config(config)

# Note that the model state is not preserved! We only saved the architecture.
new_predictions = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions - new_predictions)) > 0.

"""You can alternatively use `to_json()` from `from_json()`, which uses a JSON string to store the config instead of a Python dict. This is useful to save the config to disk."""

json_config = model.to_json()
reinitialized_model = keras.models.model_from_json(json_config)

"""### Weights-only saving

Sometimes, you are only interested in the state of the model -- its weights values -- and not in the architecture. In this case, you can retrieve the weights values as a list of Numpy arrays via `get_weights()`, and set the state of the model via `set_weights`:
"""

weights = model.get_weights()  # Retrieves the state of the model.
model.set_weights(weights)  # Sets the state of the model.

"""You can combine `get_config()`/`from_config()` and `get_weights()`/`set_weights()` to recreate your model in the same state. However, unlike `model.save()`, this will not include the training config and the optimizer. You would have to call `compile()` again before using the model for training."""

config = model.get_config()
weights = model.get_weights()

new_model = keras.Model.from_config(config)
new_model.set_weights(weights)

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)

# Note that the optimizer was not preserved,
# so the model should be compiled anew before training
# (and the optimizer will start from a blank state).

"""The save-to-disk alternative to `get_weights()` and `set_weights(weights)`
is `save_weights(fpath)` and `load_weights(fpath)`.

Here's an example that saves to disk:
"""

# Save JSON config to disk
json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
  json_file.write(json_config)
# Save weights to disk
model.save_weights('path_to_my_weights.h5')

# Reload the model from the 2 files we saved
with open('model_config.json') as json_file:
  json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)

# Note that the optimizer was not preserved.

"""But remember that the simplest, recommended way is just this:"""

model.save('path_to_my_model.h5')
del model
model = keras.models.load_model('path_to_my_model.h5')

"""### Weights-only saving in SavedModel format

Note that `save_weights` can create files either in the Keras HDF5 format,
or in the TensorFlow SavedModel format. The format is infered from the file extension
you provide: if it is ".h5" or ".keras", the framework uses the Keras HDF5 format. Anything
else defaults to SavedModel.
"""

model.save_weights('path_to_my_tf_savedmodel')

"""For total explicitness, the format can be explicitly passed via the `save_format` argument, which can take the value "tf" or "h5":"""

model.save_weights('path_to_my_tf_savedmodel', save_format='tf')

"""## Saving Subclassed Models

Sequential models and Functional models are datastructures that represent a DAG of layers. As such,
they can be safely serialized and deserialized.

A subclassed model differs in that it's not a datastructure, it's a piece of code. The architecture of the model
is defined via the body of the `call` method. This means that the architecture of the model cannot be safely serialized. To load a model, you'll need to have access to the code that created it (the code of the model subclass). Alternatively, you could be serializing this code as bytecode (e.g. via pickling), but that's unsafe and generally not portable.

For more information about these differences, see the article ["What are Symbolic and Imperative APIs in TensorFlow 2.0?"](https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021).

Let's consider the following subclassed model, which follows the same structure as the model from the first section:
"""


class ThreeLayerMLP(keras.Model):

  def __init__(self, name=None):
    super(ThreeLayerMLP, self).__init__(name=name)
    self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
    self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
    self.pred_layer = layers.Dense(10, activation='softmax', name='predictions')

  def call(self, inputs):
    x = self.dense_1(inputs)
    x = self.dense_2(x)
    return self.pred_layer(x)


def get_model():
  return ThreeLayerMLP(name='3_layer_mlp')


model = get_model()

"""First of all, *a subclassed model that has never been used cannot be saved*.

That's because a subclassed model needs to be called on some data in order to create its weights.

Until the model has been called, it does not know the shape and dtype of the input data it should be
expecting, and thus cannot create its weight variables. You may remember that in the Functional model from the first section, the shape and dtype of the inputs was specified in advance (via `keras.Input(...)`) -- that's why Functional models have a state as soon as they're instantiated.

Let's train the model, so as to give it a state:
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

"""The recommended way to save a subclassed model is to use `save_weights` to create a TensorFlow SavedModel checkpoint, which will contain the value of all variables associated with the model:
- The layers' weights
- The optimizer's state
- Any variables associated with stateful model metrics (if any)
"""

model.save_weights('path_to_my_weights', save_format='tf')

# Save predictions for future checks
predictions = model.predict(x_test)
# Also save the loss on the first batch
# to later assert that the optimizer state was preserved
first_batch_loss = model.train_on_batch(x_train[:64], y_train[:64])

"""To restore your model, you will need access to the code that created the model object.

Note that in order to restore the optimizer state and the state of any stateful  metric, you should
compile the model (with the exact same arguments as before) and call it on some data before calling `load_weights`:
"""

# Recreate the model
new_model = get_model()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())

# This initializes the variables used by the optimizers,
# as well as any stateful metric variables
new_model.train_on_batch(x_train[:1], y_train[:1])

# Load the state of the old model
new_model.load_weights('path_to_my_weights')

# Check that the model state has been preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)

# The optimizer state is preserved as well,
# so you can resume training where you left off
new_first_batch_loss = new_model.train_on_batch(x_train[:64], y_train[:64])
assert first_batch_loss == new_first_batch_loss

"""You've reached the end of this guide! This covers everything you need to know about saving and serializing models with tf.keras in TensorFlow 2.0."""
