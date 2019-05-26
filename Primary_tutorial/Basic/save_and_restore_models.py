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
import time
import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

"""### Define a model

Let's build a simple model we'll use to demonstrate saving and loading weights.
"""

# Returns a short sequential model


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()
model.summary()

"""## Save checkpoints during training

The primary use case is to automatically save checkpoints *during* and at *the end* of training. This way you can use a trained model without having to retrain it, or pick-up training where you left of—in case the training process was interrupted.

`tf.keras.callbacks.ModelCheckpoint` is a callback that performs this task. The callback takes a couple of arguments to configure checkpointing.

### Checkpoint callback usage

Train the model and pass it the `ModelCheckpoint` callback:
"""

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

"""This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:"""


"""Create a new, untrained model. When restoring a model from only weights, you must have a model with the same architecture as the original model. Since it's the same model architecture, we can share weights despite that it's a different *instance* of the model.

Now rebuild a fresh, untrained model, and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy):
"""

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

"""Then load the weights from the checkpoint, and re-evaluate:"""

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

"""### Checkpoint callback options

The callback provides several options to give the resulting checkpoints unique names, and adjust the checkpointing frequency.

Train a new model, and save uniquely named checkpoints once every 5-epochs:
"""

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

"""Now, look at the resulting checkpoints and choose the latest one:"""


latest = tf.train.latest_checkpoint(checkpoint_dir)

"""Note: the default tensorflow format only saves the 5 most recent checkpoints.

To test, reset the model and load the latest checkpoint:
"""

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

"""## What are these files?

The above code stores the weights to a collection of [checkpoint](https://www.tensorflow.org/guide/saved_model#save_and_restore_variables)-formatted files that contain only the trained weights in a binary format. Checkpoints contain:
* One or more shards that contain your model's weights.
* An index file that indicates which weights are stored in a which shard.

If you are only training a model on a single machine, you'll have one shard with the suffix: `.data-00000-of-00001`

## Manually save weights

Above you saw how to load the weights into a model.

Manually saving the weights is just as simple, use the `Model.save_weights` method.
"""

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

"""## Save the entire model

The model and optimizer can be saved to a file that contains both their state (weights and variables), and the model configuration. This allows you to export a model so it can be used without access to the original python code. Since the optimizer-state is recovered you can even resume training from exactly where you left off.

Saving a fully-functional model is very useful—you can load them in TensorFlow.js ([HDF5](https://js.tensorflow.org/tutorials/import-keras.html), [Saved Model](https://js.tensorflow.org/tutorials/import-saved-model.html)) and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite ([HDF5](https://www.tensorflow.org/lite/convert/python_api#exporting_a_tfkeras_file_), [Saved Model](https://www.tensorflow.org/lite/convert/python_api#exporting_a_savedmodel_))

### As an HDF5 file

Keras provides a basic save format using the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) standard. For our purposes, the saved model can be treated as a single binary blob.
"""

model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

"""Now recreate the model from that file:"""

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

"""Check its accuracy:"""

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

"""This technique saves everything:

* The weight values
* The model's configuration(architecture)
* The optimizer configuration

Keras saves models by inspecting the architecture. Currently, it is not able to save TensorFlow optimizers (from `tf.train`). When using those you will need to re-compile the model after loading, and you will lose the state of the optimizer.

### As a `saved_model`

Caution: This method of saving a `tf.keras` model is experimental and may change in future versions.

Build a fresh model:
"""

model = create_model()

model.fit(train_images, train_labels, epochs=5)

"""Create a `saved_model`, and place it in a time-stamped directory:"""

saved_model_path = "./saved_models/{}".format(int(time.time()))

tf.keras.experimental.export_saved_model(model, saved_model_path)

"""Reload a fresh keras model from the saved model."""

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()

# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=model.optimizer,  # keep the optimizer that was loaded
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

"""## What's Next

That was a quick guide to saving and loading in with `tf.keras`.

* The [tf.keras guide](https://www.tensorflow.org/guide/keras) shows more about saving and loading models with `tf.keras`.

* See [Saving in eager](https://www.tensorflow.org/guide/eager#object_based_saving) for saving during eager execution.

* The [Save and Restore](https://www.tensorflow.org/guide/saved_model) guide has low-level details about TensorFlow saving.
"""
