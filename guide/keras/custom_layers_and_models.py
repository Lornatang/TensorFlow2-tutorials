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
from tensorflow.python.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

"""## The Layer class

### Layers encapsulate a state (weights) and some computation

The main data structure you'll work with is the `Layer`.
A layer encapsulates both a state (the layer's "weights")
and a transformation from inputs to outputs (a "call", the layer's
forward pass).

Here's a densely-connected layer. It has a state: the variables `w` and `b`.
"""




class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)

"""Note that the weights `w` and `b` are automatically tracked by the layer upon
being set as layer attributes:
"""

assert linear_layer.weights == [linear_layer.w, linear_layer.b]

"""Note you also have access to a quicker shortcut for adding weight to a layer: the `add_weight` method:"""


class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(units,),
                             initializer='zeros',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)

"""#### Layers can have non-trainable weights

Besides trainable weights, you can add non-trainable weights to a layer as well.
Such weights are meant not to be taken into account during backpropagation,
when you are training the layer.

Here's how to add and use a non-trainable weight:
"""


class ComputeSum(layers.Layer):

  def __init__(self, input_dim):
    super(ComputeSum, self).__init__()
    self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                             trainable=False)

  def call(self, inputs):
    self.total.assign_add(tf.reduce_sum(inputs, axis=0))
    return self.total


x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())

"""It's part of `layer.weights`, but it gets categorized as a non-trainable weight:"""

print('weights:', len(my_sum.weights))
print('non-trainable weights:', len(my_sum.non_trainable_weights))

# It's not included in the trainable weights:
print('trainable_weights:', my_sum.trainable_weights)

"""### Best practice: deferring weight creation until the shape of the inputs is known

In the logistic regression example above, our `Linear` layer took an `input_dim` argument
that was used to compute the shape of the weights `w` and `b` in `__init__`:
"""


class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(units,),
                             initializer='zeros',
                             trainable=True)


"""In many cases, you may not know in advance the size of your inputs, and you would
like to lazily create weights when that value becomes known,
some time after instantiating the layer.

In the Keras API, we recommend creating layer weights in the `build(inputs_shape)` method of your layer.
Like this:
"""


class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


"""The `__call__` method of your layer will automatically run `build` the first time it is called.
You now have a layer that's lazy and easy to use:
"""

linear_layer = Linear(32)  # At instantiation, we don't know on what inputs this is going to get called
y = linear_layer(x)  # The layer's weights are created dynamically the first time the layer is called

"""### Layers are recursively composable

If you assign a Layer instance as attribute of another Layer,
the outer layer will start tracking the weights of the inner layer.

We recommend creating such sublayers in the `__init__` method (since the sublayers will typically have a `build` method, they will be built when the outer layer gets built).
"""


# Let's assume we are reusing the Linear class
# with a `build` method that we defined above.

class MLPBlock(layers.Layer):

  def __init__(self):
    super(MLPBlock, self).__init__()
    self.linear_1 = Linear(32)
    self.linear_2 = Linear(32)
    self.linear_3 = Linear(1)

  def call(self, inputs):
    x = self.linear_1(inputs)
    x = tf.nn.relu(x)
    x = self.linear_2(x)
    x = tf.nn.relu(x)
    return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))

"""### Layers recursively collect losses created during the forward pass

When writing the `call` method of a layer, you can create loss tensors that you will want to use later, when writing your training loop. This is doable by calling `self.add_loss(value)`:
"""


# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(layers.Layer):

  def __init__(self, rate=1e-2):
    super(ActivityRegularizationLayer, self).__init__()
    self.rate = rate

  def call(self, inputs):
    self.add_loss(self.rate * tf.reduce_sum(inputs))
    return inputs


"""These losses (including those created by any inner layer) can be retrieved via `layer.losses`.
This property is reset at the start of every `__call__` to the top-level layer, so that `layer.losses` always contains the loss values created during the last forward pass.
"""


class OuterLayer(layers.Layer):

  def __init__(self):
    super(OuterLayer, self).__init__()
    self.activity_reg = ActivityRegularizationLayer(1e-2)

  def call(self, inputs):
    return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # No losses yet since the layer has never been called
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # We created one loss value

# `layer.losses` gets reset at the start of each __call__
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above

"""In addition, the `loss` property also contains regularization losses created for the weights of any inner layer:"""


class OuterLayer(layers.Layer):

  def __init__(self):
    super(OuterLayer, self).__init__()
    self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

  def call(self, inputs):
    return self.dense(inputs)


layer = OuterLayer()
_ = layer(tf.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel)`,
# created by the `kernel_regularizer` above.
print(layer.losses)

"""These losses are meant to be taken into account when writing training loops, like this:


```python
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Iterate over the batches of a dataset.
for x_batch_train, y_batch_train in train_dataset:
  with tf.GradientTape() as tape:
    logits = layer(x_batch_train)  # Logits for this minibatch
    # Loss value for this minibatch
    loss_value = loss_fn(y_batch_train, logits)
    # Add extra losses created during this forward pass:
    loss_value += sum(model.losses)

  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

For a detailed guide about writing training loops, see the second section of the [Guide to Training & Evaluation](./training_and_evaluation.ipynb).

### You can optionally enable serialization on your layers

If you need your custom layers to be serializable as part of a [Functional model](./functional.ipynb), you can optionally implement a `get_config` method:
"""


class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_config(self):
    return {'units': self.units}


# Now you can recreate the layer from its config:
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

"""Note that the `__init__` method of the base `Layer` class takes some keyword arguments, in particular a `name` and a `dtype`. It's good practice to pass these arguments to the parent class in `__init__` and to include them in the layer config:"""


class Linear(layers.Layer):

  def __init__(self, units=32, **kwargs):
    super(Linear, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_config(self):
    config = super(Linear, self).get_config()
    config.update({'units': self.units})
    return config


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

"""If you need more flexibility when deserializing the layer from its config, you can also override the `from_config` class method. This is the base implementation of `from_config`:

```python
def from_config(cls, config):
  return cls(**config)
```

To learn more about serialization and saving, see the complete [Guide to Saving and Serializing Models](./saving_and_serializing.ipynb).

### Privileged `training` argument in the `call` method


Some layers, in particular the `BatchNormalization` layer and the `Dropout` layer, have different behaviors during training and inference. For such layers, it is standard practice to expose a `training` (boolean) argument in the `call` method.

By exposing this argument in `call`, you enable the built-in training and evaluation loops (e.g. `fit`) to correctly use the layer in training and inference.
"""


class CustomDropout(layers.Layer):

  def __init__(self, rate, **kwargs):
    super(CustomDropout, self).__init__(**kwargs)
    self.rate = rate

  def call(self, inputs, training=None):
    if training:
      return tf.nn.dropout(inputs, rate=self.rate)
    return inputs


"""## Building Models

### The Model class

In general, you will use the `Layer` class to define inner computation blocks,
and will use the `Model` class to define the outer model -- the object you will train.

For instance, in a ResNet50 model, you would have several ResNet blocks subclassing `Layer`,
and a single `Model` encompassing the entire ResNet50 network.

The `Model` class has the same API as `Layer`, with the following differences:
- It exposes built-in training, evaluation, and prediction loops (`model.fit()`, `model.evaluate()`, `model.predict()`).
- It exposes the list of its inner layers, via the `model.layers` property.
- It exposes saving and serialization APIs.

Effectively, the "Layer" class corresponds to what we refer to in the literature
as a "layer" (as in "convolution layer" or "recurrent layer") or as a "block" (as in "ResNet block" or "Inception block").

Meanwhile, the "Model" class corresponds to what is referred to in the literature
as a "model" (as in "deep learning model") or as a "network" (as in "deep neural network").

For instance, we could take our mini-resnet example above, and use it to build a `Model` that we could
train with `fit()`, and that we could save with `save_weights`:

```python
class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save_weights(filepath)
```

### Putting it all together: an end-to-end example

Here's what you've learned so far:
- A `Layer` encapsulate a state (created in `__init__` or `build`) and some computation (in `call`).
- Layers can be recursively nested to create new, bigger computation blocks.
- Layers can create and track losses (typically regularization losses).
- The outer container, the thing you want to train, is a `Model`. A `Model` is just like a `Layer`, but with added training and serialization utilities.

Let's put all of these things together into an end-to-end example: we're going to implement a Variational AutoEncoder (VAE). We'll train it on MNIST digits.

Our VAE will be a subclass of `Model`, built as a nested composition of layers that subclass `Layer`. It will feature a regularization loss (KL divergence).
"""


class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
               latent_dim=32,
               intermediate_dim=64,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.dense_proj(inputs)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               name='decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_output = layers.Dense(original_dim, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_proj(inputs)
    return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               latent_dim=32,
               name='autoencoder',
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.original_dim = original_dim
    self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
    self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    kl_loss = - 0.5 * tf.reduce_mean(
      z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    self.add_loss(kl_loss)
    return reconstructed


original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Iterate over epochs.
for epoch in range(3):
  print('Start of epoch %d' % (epoch,))

  # Iterate over the batches of the dataset.
  for step, x_batch_train in enumerate(train_dataset):
    with tf.GradientTape() as tape:
      reconstructed = vae(x_batch_train)
      # Compute reconstruction loss
      loss = mse_loss_fn(x_batch_train, reconstructed)
      loss += sum(vae.losses)  # Add KLD regularization loss

    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    loss_metric(loss)

    if step % 100 == 0:
      print('step %s: mean loss = %s' % (step, loss_metric.result()))

"""Note that since the VAE is subclassing `Model`, it features built-in training loops. So you could also have trained it like this:"""

vae = VariationalAutoEncoder(784, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)

"""### Beyond object-oriented development: the Functional API

Was this example too much object-oriented development for you? You can also build models using [the Functional API](./functional.ipynb). Importantly, choosing one style or another does not prevent you from leveraging components written in the other style: you can always mix-and-match.

For instance, the Functional API example below reuses the same `Sampling` layer we defined in the example above.
"""

original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
x = layers.Dense(intermediate_dim, activation='relu')(original_inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')

# Add KL divergence regularization loss.
kl_loss = - 0.5 * tf.reduce_mean(
  z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
