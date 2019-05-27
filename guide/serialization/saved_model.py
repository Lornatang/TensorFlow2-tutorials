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
from matplotlib import pyplot as plt
import numpy as np

file = tf.keras.utils.get_file(
  "grace_hopper.jpg",
  "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
  x[tf.newaxis, ...])

"""We'll use an image of Grace Hopper as a running example, and a Keras pre-trained image classification model since it's easy to use. Custom models work too, and are covered in detail later."""

# tf.keras.applications.vgg19.decode_predictions
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                      'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

pretrained_model = tf.keras.applications.MobileNet()
result_before_save = pretrained_model(x)
print()

decoded = imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5] + 1]

print("Result before saving:\n", decoded)

"""The top prediction for this image is "military uniform"."""

tf.saved_model.save(pretrained_model, "/tmp/mobilenet/1/")

"""The save-path follows a convention used by TensorFlow Serving where the last path component (`1/` here) is a version number for your model - it allows tools like Tensorflow Serving to reason about the relative freshness.

SavedModels have named functions called signatures. Keras models export their forward pass under the `serving_default` signature key. The [SavedModel command line interface](#saved_model_cli) is useful for inspecting SavedModels on disk:
"""

"""We can load the SavedModel back into Python with `tf.saved_model.load` and see how Admiral Hopper's image is classified."""

loaded = tf.saved_model.load("/tmp/mobilenet/1/")
print(list(loaded.signatures.keys()))  # ["serving_default"]

"""Imported signatures always return dictionaries."""

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

"""Running inference from the SavedModel gives the same result as the original model."""

labeling = infer(tf.constant(x))["reshape_2"]

decoded = imagenet_labels[np.argsort(labeling)[0, ::-1][:5] + 1]

print("Result after saving and loading:\n", decoded)

"""## Serving the model

SavedModels are usable from Python, but production environments typically use a dedicated service for inference. This is easy to set up from a SavedModel using TensorFlow Serving.

See the [TensorFlow Serving REST tutorial](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/tutorials/Serving_REST_simple.ipynb) for more details about serving, including instructions for installing `tensorflow_model_server` in a notebook or on your local machine. As a quick sketch, to serve the `mobilenet` model exported above just point the model server at the SavedModel directory:

```bash
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=mobilenet \
  --model_base_path="/tmp/mobilenet" >server.log 2>&1
```

  Then send a request.

```python
!pip install requests
import json
import numpy
import requests
data = json.dumps({"signature_name": "serving_default",
                   "instances": x.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/mobilenet:predict',
                              data=data, headers=headers)
predictions = numpy.array(json.loads(json_response.text)["predictions"])
```

The resulting `predictions` are identical to the results from Python.

### SavedModel format

A SavedModel is a directory containing serialized signatures and the state needed to run them, including variable values and vocabularies.
"""

"""The `assets` directory contains files used by the TensorFlow graph, for example text files used to initialize vocabulary tables. It is unused in this example.

SavedModels may have an `assets.extra` directory for any files not used by the TensorFlow graph, for example information for consumers about what to do with the SavedModel. TensorFlow itself does not use this directory.

### Exporting custom models

In the first section, `tf.saved_model.save` automatically determined a signature for the `tf.keras.Model` object. This worked because Keras `Model` objects have an unambiguous method to export and known input shapes. `tf.saved_model.save` works just as well with low-level model building APIs, but you will need to indicate which function to use as a signature if you're planning to serve a model.
"""


class CustomModule(tf.Module):

  def __init__(self):
    super(CustomModule, self).__init__()
    self.v = tf.Variable(1.)

  @tf.function
  def __call__(self, x):
    return x * self.v

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def mutate(self, new_v):
    self.v.assign(new_v)


module = CustomModule()

"""This module has two methods decorated with `tf.function`. While these functions will be included in the SavedModel and available if the SavedModel is reloaded via `tf.saved_model.load` into a Python program, without explicitly declaring the serving signature tools like Tensorflow Serving and `saved_model_cli` cannot access them.

`module.mutate` has an `input_signature`, and so there is enough information to save its computation graph in the SavedModel already. `__call__` has no signature and so this method needs to be called before saving.
"""

module(tf.constant(0.))
tf.saved_model.save(module, "/tmp/module_no_signatures")

"""For functions without an `input_signature`, any input shapes used before saving will be available after loading. Since we called `__call__` with just a scalar, it will accept only scalar values."""

imported = tf.saved_model.load("/tmp/module_no_signatures")
assert 3. == imported(tf.constant(3.)).numpy()
imported.mutate(tf.constant(2.))
assert 6. == imported(tf.constant(3.)).numpy()

"""The function will not accept new shapes like vectors.

```python
imported(tf.constant([3.]))
```

<pre>
ValueError: Could not find matching function to call for canonicalized inputs ((<tf.Tensor 'args_0:0' shape=(1,) dtype=float32>,), {}). Only existing signatures are [((TensorSpec(shape=(), dtype=tf.float32, name=u'x'),), {})].
</pre>

`get_concrete_function` lets you add input shapes to a function without calling it. It takes `tf.TensorSpec` objects in place of `Tensor` arguments, indicating the shapes and dtypes of inputs. Shapes can either be `None`, indicating that any shape is acceptable, or a list of axis sizes. If an axis size is `None` then any size is acceptable for that axis. `tf.TensorSpecs` can also have names, which default to the function's argument keywords ("x" here).
"""

module.__call__.get_concrete_function(x=tf.TensorSpec([None], tf.float32))
tf.saved_model.save(module, "/tmp/module_no_signatures")
imported = tf.saved_model.load("/tmp/module_no_signatures")
assert [3.] == imported(tf.constant([3.])).numpy()

"""Functions and variables attached to objects like `tf.keras.Model` and `tf.Module` are available on import, but many Python types and attributes are lost. The Python program itself is not saved in the SavedModel.

We didn't identify any of the functions we exported as a signature, so it has none.
"""

"""We exported a single signature, and its key defaulted to "serving_default". To export multiple signatures, pass a dictionary."""


@tf.function(input_signature=[tf.TensorSpec([], tf.string)])
def parse_string(string_input):
  return imported(tf.strings.to_number(string_input))


signatures = {"serving_default": parse_string,
              "from_float": imported.signatures["serving_default"]}

tf.saved_model.save(imported, "/tmp/module_with_multiple_signatures", signatures)

"""## Fine-tuning imported models

Variable objects are available, and we can backprop through imported functions.
"""

optimizer = tf.optimizers.SGD(0.05)


def train_step():
  with tf.GradientTape() as tape:
    loss = (10. - imported(tf.constant(2.))) ** 2
  variables = tape.watched_variables()
  grads = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(grads, variables))
  return loss


for _ in range(10):
  # "v" approaches 5, "loss" approaches 0
  print("loss={:.2f} v={:.2f}".format(train_step(), imported.v.numpy()))

"""## Control flow in SavedModels

Anything that can go in a `tf.function` can go in a SavedModel. With [AutoGraph](./autograph.ipynb) this includes conditional logic which depends on Tensors, specified with regular Python control flow.
"""


@tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
def control_flow(x):
  if x < 0:
    tf.print("Invalid!")
  else:
    tf.print(x % 3)


to_export = tf.Module()
to_export.control_flow = control_flow
tf.saved_model.save(to_export, "/tmp/control_flow")

imported = tf.saved_model.load("/tmp/control_flow")
imported.control_flow(tf.constant(-1))  # Invalid!
imported.control_flow(tf.constant(2))  # 2
imported.control_flow(tf.constant(3))  # 0

"""## SavedModels from Estimators

Estimators export SavedModels through [`tf.Estimator.export_saved_model`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#export_saved_model). See the [guide to Estimator](https://www.tensorflow.org/guide/estimators) for details.
"""

input_column = tf.feature_column.numeric_column("x")
estimator = tf.estimator.LinearClassifier(feature_columns=[input_column])


def input_fn():
  return tf.data.Dataset.from_tensor_slices(
    ({"x": [1., 2., 3., 4.]}, [1, 1, 0, 0])).repeat(200).shuffle(64).batch(16)


estimator.train(input_fn)

serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
  tf.feature_column.make_parse_example_spec([input_column]))
export_path = estimator.export_saved_model(
  "/tmp/from_estimator/", serving_input_fn)

"""This SavedModel accepts serialized `tf.Example` protocol buffers, which are useful for serving. But we can also load it with `tf.saved_model.load` and run it from Python."""

imported = tf.saved_model.load(export_path)


def predict(x):
  example = tf.train.Example()
  example.features.feature["x"].float_list.value.extend([x])
  return imported.signatures["predict"](
    examples=tf.constant([example.SerializeToString()]))


print(predict(1.5))
print(predict(3.5))

"""`tf.estimator.export.build_raw_serving_input_receiver_fn` allows you to create input functions which take raw tensors rather than `tf.train.Example`s.

## Load a SavedModel in C++

The C++ version of the SavedModel [loader](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/loader.h) provides an API to load a SavedModel from a path, while allowing SessionOptions and RunOptions. You have to specify the tags associated with the graph to be loaded. The loaded version of SavedModel is referred to as SavedModelBundle and contains the MetaGraphDef and the session within which it is loaded.

```C++
const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &bundle);
```

<a id=saved_model_cli/>

## Details of the SavedModel command line interface

You can use the SavedModel Command Line Interface (CLI) to inspect and
execute a SavedModel.
For example, you can use the CLI to inspect the model's `SignatureDef`s.
The CLI enables you to quickly confirm that the input
Tensor dtype and shape match the model. Moreover, if you
want to test your model, you can use the CLI to do a sanity check by
passing in sample inputs in various formats (for example, Python
expressions) and then fetching the output.


### Install the SavedModel CLI

Broadly speaking, you can install TensorFlow in either of the following
two ways:

*  By installing a pre-built TensorFlow binary.
*  By building TensorFlow from source code.

If you installed TensorFlow through a pre-built TensorFlow binary,
then the SavedModel CLI is already installed on your system
at pathname `bin\saved_model_cli`.

If you built TensorFlow from source code, you must run the following
additional command to build `saved_model_cli`:

```
$ bazel build tensorflow/python/tools:saved_model_cli
```

### Overview of commands

The SavedModel CLI supports the following two commands on a
`MetaGraphDef` in a SavedModel:

* `show`, which shows a computation on a `MetaGraphDef` in a SavedModel.
* `run`, which runs a computation on a `MetaGraphDef`.


### `show` command

A SavedModel contains one or more `MetaGraphDef`s, identified by their tag-sets.
To serve a model, you
might wonder what kind of `SignatureDef`s are in each model, and what are their
inputs and outputs.  The `show` command let you examine the contents of the
SavedModel in hierarchical order.  Here's the syntax:

```
usage: saved_model_cli show [-h] --dir DIR [--all]
[--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
```

For example, the following command shows all available
MetaGraphDef tag-sets in the SavedModel:

```
$ saved_model_cli show --dir /tmp/saved_model_dir
The given SavedModel contains the following tag-sets:
serve
serve, gpu
```

The following command shows all available `SignatureDef` keys in
a `MetaGraphDef`:

```
$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
following keys:
SignatureDef key: "classify_x2_to_y3"
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"
```

If a `MetaGraphDef` has *multiple* tags in the tag-set, you must specify
all tags, each tag separated by a comma. For example:

<pre>
$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
</pre>

To show all inputs and outputs TensorInfo for a specific `SignatureDef`, pass in
the `SignatureDef` key to `signature_def` option. This is very useful when you
want to know the tensor key value, dtype and shape of the input tensors for
executing the computation graph later. For example:

```
$ saved_model_cli show --dir \
/tmp/saved_model_dir --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['x'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: x:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['y'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: y:0
Method name is: tensorflow/serving/predict
```

To show all available information in the SavedModel, use the `--all` option.
For example:

<pre>
$ saved_model_cli show --dir /tmp/saved_model_dir --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/classify

...

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/predict
</pre>


### `run` command

Invoke the `run` command to run a graph computation, passing
inputs and then displaying (and optionally saving) the outputs.
Here's the syntax:

```
usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
                           SIGNATURE_DEF_KEY [--inputs INPUTS]
                           [--input_exprs INPUT_EXPRS]
                           [--input_examples INPUT_EXAMPLES] [--outdir OUTDIR]
                           [--overwrite] [--tf_debug]
```

The `run` command provides the following three ways to pass inputs to the model:

* `--inputs` option enables you to pass numpy ndarray in files.
* `--input_exprs` option enables you to pass Python expressions.
* `--input_examples` option enables you to pass `tf.train.Example`.

#### `--inputs`

To pass input data in files, specify the `--inputs` option, which takes the
following general format:

```bsh
--inputs <INPUTS>
```

where *INPUTS* is either of the following formats:

*  `<input_key>=<filename>`
*  `<input_key>=<filename>[<variable_name>]`

You may pass multiple *INPUTS*. If you do pass multiple inputs, use a semicolon
to separate each of the *INPUTS*.

`saved_model_cli` uses `numpy.load` to load the *filename*.
The *filename* may be in any of the following formats:

*  `.npy`
*  `.npz`
*  pickle format

A `.npy` file always contains a numpy ndarray. Therefore, when loading from
a `.npy` file, the content will be directly assigned to the specified input
tensor. If you specify a *variable_name* with that `.npy` file, the
*variable_name* will be ignored and a warning will be issued.

When loading from a `.npz` (zip) file, you may optionally specify a
*variable_name* to identify the variable within the zip file to load for
the input tensor key.  If you don't specify a *variable_name*, the SavedModel
CLI will check that only one file is included in the zip file and load it
for the specified input tensor key.

When loading from a pickle file, if no `variable_name` is specified in the
square brackets, whatever that is inside the pickle file will be passed to the
specified input tensor key. Otherwise, the SavedModel CLI will assume a
dictionary is stored in the pickle file and the value corresponding to
the *variable_name* will be used.


#### `--input_exprs`

To pass inputs through Python expressions, specify the `--input_exprs` option.
This can be useful for when you don't have data
files lying around, but still want to sanity check the model with some simple
inputs that match the dtype and shape of the model's `SignatureDef`s.
For example:

```bsh
`<input_key>=[[1],[2],[3]]`
```

In addition to Python expressions, you may also pass numpy functions. For
example:

```bsh
`<input_key>=np.ones((32,32,3))`
```

(Note that the `numpy` module is already available to you as `np`.)


#### `--input_examples`

To pass `tf.train.Example` as inputs, specify the `--input_examples` option.
For each input key, it takes a list of dictionary, where each dictionary is an
instance of `tf.train.Example`. The dictionary keys are the features and the
values are the value lists for each feature.
For example:

```bsh
`<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`
```

#### Save output

By default, the SavedModel CLI writes output to stdout. If a directory is
passed to `--outdir` option, the outputs will be saved as `.npy` files named after
output tensor keys under the given directory.

Use `--overwrite` to overwrite existing output files.
"""
