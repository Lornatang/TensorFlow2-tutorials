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

# Import TensorFlow
import tensorflow as tf

import numpy as np

"""## Types of strategies
`tf.distribute.Strategy` intends to cover a number of use cases along different axes. Some of these combinations are currently supported and others will be added in the future. Some of these axes are:

* Syncronous vs asynchronous training: These are two common ways of distributing training with data parallelism. In sync training, all workers train over different slices of input data in sync, and aggregating gradients at each step. In async training, all workers are independently training over the input data and updating variables asynchronously. Typically sync training is supported via all-reduce and async through parameter server architecture.
* Hardware platform: Users may want to scale their training onto multiple GPUs on one machine, or multiple machines in a network (with 0 or more GPUs each), or on Cloud TPUs.

In order to support these use cases, we have 4 strategies available. In the next section we will talk about which of these are supported in which scenarios in TF 2.0-alpha at this time.

### MirroredStrategy
`tf.distribute.MirroredStrategy` support synchronous distributed training on multiple GPUs on one machine. It creates one replica per GPU device. Each variable in the model is mirrored across all the replicas. Together, these variables form a single conceptual variable called `MirroredVariable`. These variables are kept in sync with each other by applying identical updates.

Efficient all-reduce algorithms are used to communicate the variable updates across the devices.
All-reduce aggregates tensors across all the devices by adding them up, and makes them available on each device.
It’s a fused algorithm that is very efficient and can reduce the overhead of synchronization significantly. There are many all-reduce algorithms and implementations available, depending on the type of communication available between devices. By default, it uses NVIDIA NCCL as the all-reduce implementation. The user can also choose between a few other options we provide, or write their own.

Here is the simplest way of creating `MirroredStrategy`:
"""

mirrored_strategy = tf.distribute.MirroredStrategy()

"""This will create a `MirroredStrategy` instance which will use all the GPUs that are visible to TensorFlow, and use NCCL as the cross device communication.

If you wish to use only some of the GPUs on your machine, you can do so like this:
"""

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

"""If you wish to override the cross device communication, you can do so using the `cross_device_ops` argument by supplying an instance of `tf.distribute.CrossDeviceOps`. Currently we provide `tf.distribute.HierarchicalCopyAllReduce` and `tf.distribute.ReductionToOneDevice` as 2 other options other than `tf.distribute.NcclAllReduce` which is the default."""

mirrored_strategy = tf.distribute.MirroredStrategy(
  cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

"""### CentralStorageStrategy
`tf.distribute.experimental.CentralStorageStrategy` does synchronous training as well. Variables are not mirrored, instead they are placed on the CPU and operations are replicated across all local GPUs. If there is only one GPU, all variables and operations will be placed on that GPU.

Create a `CentralStorageStrategy` by:
"""

central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()

"""This will create a `CentralStorageStrategy` instance which will use all visible GPUs and CPU. Update to variables on replicas will be aggragated before being applied to variables.

Note: This strategy is [`experimental`](https://www.tensorflow.org/guide/version_compat#what_is_not_covered) as we are currently improving it and making it work for more scenarios. As part of this, please expect the APIs to change in the future.

### MultiWorkerMirroredStrategy

`tf.distribute.experimental.MultiWorkerMirroredStrategy` is very similar to `MirroredStrategy`. It implements synchronous distributed training across multiple workers, each with potentially multiple GPUs. Similar to `MirroredStrategy`, it creates copies of all variables in the model on each device across all workers.

It uses [CollectiveOps](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/collective_ops.py) as the multi-worker all-reduce communication method used to keep variables in sync. A collective op is a single op in the TensorFlow graph which can automatically choose an all-reduce algorithm in the TensorFlow runtime according to hardware, network topology and tensor sizes.

It also implements additional performance optimizations. For example, it includes a static optimization that converts multiple all-reductions on small tensors into fewer all-reductions on larger tensors. In addition, we are designing it to have a plugin architecture - so that in the future, users will be able to plugin algorithms that are better tuned for their hardware. Note that collective ops also implement other collective operations such as broadcast and all-gather.

Here is the simplest way of creating `MultiWorkerMirroredStrategy`:
"""

multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

"""`MultiWorkerMirroredStrategy` currently allows you to choose between two different implementations of collective ops.  `CollectiveCommunication.RING` implements ring-based collectives using gRPC as the communication layer.  `CollectiveCommunication.NCCL` uses [Nvidia's NCCL](https://developer.nvidia.com/nccl) to implement collectives.  `CollectiveCommunication.AUTO` defers the choice to the runtime.  The best choice of collective implementation depends upon the number and kind of GPUs, and the network interconnect in the cluster. You can specify them like so:"""

multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
  tf.distribute.experimental.CollectiveCommunication.NCCL)

"""One of the key differences to get multi worker training going, as compared to multi-GPU training, is the multi-worker setup. "TF_CONFIG" environment variable is the standard way in TensorFlow to specify the cluster configuration to each worker that is part of the cluster. See section on ["TF_CONFIG" below](#TF_CONFIG) for more details on how this can be done.

Note: This strategy is [`experimental`](https://www.tensorflow.org/guide/version_compat#what_is_not_covered) as we are currently improving it and making it work for more scenarios. As part of this, please expect the APIs to change in the future.

### TPUStrategy
`tf.distribute.experimental.TPUStrategy` lets users run their TensorFlow training on Tensor Processing Units (TPUs). TPUs are Google's specialized ASICs designed to dramatically accelerate machine learning workloads. They are available on Google Colab, the [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) and [Google Compute Engine](https://cloud.google.com/tpu).

In terms of distributed training architecture, TPUStrategy is the same `MirroredStrategy` - it implements synchronous distributed training. TPUs provide their own implementation of efficient all-reduce and other collective operations across multiple TPU cores, which are used in `TPUStrategy`.

Here is how you would instantiate `TPUStrategy`.
Note: To run this code in Colab, you should select TPU as the Colab runtime. See [Using TPUs]( tpu.ipynb) guide for a runnable version.

```
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)
```

`TPUClusterResolver` instance helps locate the TPUs. In Colab, you don't need to specify any arguments to it. If you want to use this for Cloud TPUs, you will need to specify the name of your TPU resource in `tpu` argument. We also need to initialize the tpu system explicitly at the start of the program. This is required before TPUs can be used for computation and should ideally be done at the beginning because it also wipes out the TPU memory so all state will be lost.

Note: This strategy is [`experimental`](https://www.tensorflow.org/guide/version_compat#what_is_not_covered) as we are currently improving it and making it work for more scenarios. As part of this, please expect the APIs to change in the future.

### ParameterServerStrategy
`tf.distribute.experimental.ParameterServerStrategy` supports parameter servers training on multiple machines. In this setup, some machines are designated as workers and some as parameter servers. Each variable of the model is placed on one parameter server. Computation is replicated across all GPUs of the all the workers.

In terms of code, it looks similar to other strategies:
```
ps_strategy = tf.distribute.experimental.ParameterServerStrategy()
```

For multi worker training, "TF_CONFIG" needs to specify the configuration of parameter servers and workers in your cluster, which you can read more about in ["TF_CONFIG" below](#TF_CONFIG) below.

So far we've talked about what are the different stategies available and how you can instantiate them. In the next few sections, we will talk about the different ways in which you can use them to distribute your training. We will show short code snippets in this guide and link off to full tutorials which you can run end to end.

## Using `tf.distribute.Strategy` with Keras
We've integrated `tf.distribute.Strategy` into `tf.keras` which is TensorFlow's implementation of the
[Keras API specification](https://keras.io). `tf.keras`  is a high-level API to build and train models. By integrating into `tf.keras` backend, we've made it seamless for Keras users to distribute their training written in the Keras training framework. The only things that need to change in a user's program are: (1) Create an instance of the appropriate `tf.distribute.Strategy` and (2) Move the creation and compiling of Keras model inside `strategy.scope`.

Here is a snippet of code to do this for a very simple Keras model with one dense layer:
"""

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  model.compile(loss='mse', optimizer='sgd')

"""In this example we used `MirroredStrategy` so we can run this on a machine with multiple GPUs. `strategy.scope()` indicated which parts of the code to run distributed. Creating a model inside this scope allows us to create mirrored variables instead of regular variables. Compiling under the scope allows us to know that the user intends to train this model using this strategy. Once this is setup, you can fit your model like you would normally. `MirroredStrategy` takes care of replicating the model's training on the available GPUs, aggregating gradients etc."""

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)

"""Here we used a `tf.data.Dataset` to provide the training and eval input. You can also use numpy arrays:"""

inputs, targets = np.ones((100, 1)), np.ones((100, 1))
model.fit(inputs, targets, epochs=2, batch_size=10)

"""In both cases (dataset or numpy), each batch of the given input is divided equally among the multiple replicas. For instance, if using `MirroredStrategy` with 2 GPUs, each batch of size 10 will get divided among the 2 GPUs, with each receiving 5 input examples in each step. Each epoch will then train faster as you add more GPUs. Typically, you would want to increase your batch size as you add more accelerators so as to make effective use of the extra computing power. You will also need to re-tune your learning rate, depending on the model. You can use `strategy.num_replicas_in_sync` to get the number of replicas."""

# Compute global batch size using number of replicas.
BATCH_SIZE_PER_REPLICA = 5
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100)
dataset = dataset.batch(global_batch_size)

LEARNING_RATES_BY_BATCH_SIZE = {5: 0.1, 10: 0.15}
learning_rate = LEARNING_RATES_BY_BATCH_SIZE[global_batch_size]

"""### What's supported now?

In TF 2.0 alpha release, we support training with Keras using `MirroredStrategy`, as well as one machine parameter server using `ParameterServerStrategy`.
Support for other strategies will be coming soon. The API and how to use will be exactly the same as above. If you wish to use the other strategies like `TPUStrategy` or `MultiWorkerMirorredStrategy` in Keras in TF 2.0, you can currently do so by disabling eager execution (`tf.compat.v1.disable_eager_execution()`).

### Examples and Tutorials

Here is a list of tutorials and examples that illustrate the above integration end to end with Keras:

1. [Tutorial](../tutorials/distribute/keras.ipynb) to train MNIST with `MirroredStrategy`.
2. Official [ResNet50](https://github.com/tensorflow/models/blob/master/official/resnet/keras/keras_imagenet_main.py) training with ImageNet data using `MirroredStrategy`.
3. [ResNet50](https://github.com/tensorflow/tpu/blob/master/models/experimental/resnet50_keras/resnet50.py) trained with Imagenet data on Cloud TPus with `TPUStrategy`. Note that this example only works with TensorFlow 1.x currently.

## Using `tf.distribute.Strategy` with Estimator
`tf.estimator` is a distributed training TensorFlow API that originally supported the async parameter server approach. Like with Keras, we've integrated `tf.distribute.Strategy` into `tf.Estimator` so that a user who is using Estimator for their training can easily change their training is distributed with very few changes to your their code. With this, estimator users can now do synchronous distributed training on multiple GPUs and multiple workers, as well as use TPUs.

The usage of `tf.distribute.Strategy` with Estimator is slightly different than the Keras case. Instead of using `strategy.scope`, now we pass the strategy object into the [`RunConfig`](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig) for the Estimator.

Here is a snippet of code that shows this with a premade estimator `LinearRegressor` and `MirroredStrategy`:
"""

mirrored_strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
  train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)
regressor = tf.estimator.LinearRegressor(
  feature_columns=[tf.feature_column.numeric_column('feats')],
  optimizer='SGD',
  config=config)

"""We use a premade Estimator here, but the same code works with a custom Estimator as well. `train_distribute` determines how training will be distributed, and `eval_distribute` determines how evaluation will be distributed. This is another difference from Keras where we use the same strategy for both training and eval.

Now we can train and evaluate this Estimator with an input function:
"""


def input_fn():
  dataset = tf.data.Dataset.from_tensors(({"feats": [1.]}, [1.]))
  return dataset.repeat(1000).batch(10)


regressor.train(input_fn=input_fn, steps=10)
regressor.evaluate(input_fn=input_fn, steps=10)

"""Another difference to highlight here between Estimator and Keras is the input handling. In Keras, we mentioned that each batch of the dataset is split across the multiple replicas. In Estimator, however, the user provides an `input_fn` and have full control over how they want their data to be distributed across workers and devices. We do not do automatic splitting of batch, nor automatically shard the data across different workers. The provided `input_fn` is called once per worker, thus giving one dataset per worker. Then one batch from that dataset is fed to one replica on that worker, thereby consuming N batches for N replicas on 1 worker. In other words, the dataset returned by the `input_fn` should provide batches of size `PER_REPLICA_BATCH_SIZE`. And the global batch size for a step can be obtained as `PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync`. When doing multi worker training, users will also want to either split their data across the workers, or shuffle with a random seed on each. You can see an example of how to do this in the [multi-worker tutorial](../tutorials/distribute/multi_worker.ipynb).

We showed an example of using `MirroredStrategy` with Estimator. You can also use `TPUStrategy` with Estimator as well, in the exact same way:
```
config = tf.estimator.RunConfig(
    train_distribute=tpu_strategy, eval_distribute=tpu_strategy)
```

And similarly, you can use multi worker and parameter server strategies as well. The code remains the same, but you need to use `tf.estimator.train_and_evaluate`, and set "TF_CONFIG" environment variables for each binary running in your cluster.

### What's supported now?

In TF 2.0 alpha release, we support training with Estimator using all strategies.

### Examples and Tutorials
Here are some examples that show end to end usage of various strategies with Estimator:

1. [Tutorial](../tutorials/distribute/multi_worker.ipynb) to train MNIST with multiple workers using `MultiWorkerMirroredStrategy`.
2. [End to end example](https://github.com/tensorflow/ecosystem/tree/master/distribution_strategy) for multi worker training in tensorflow/ecosystem using Kuberentes templates. This example starts with a Keras model and converts it to an Estimator using the `tf.keras.estimator.model_to_estimator` API.
3. Official [ResNet50](https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_main.py) model, which can be trained using either `MirroredStrategy` or `MultiWorkerMirroredStrategy`.
4. [ResNet50](https://github.com/tensorflow/tpu/blob/master/models/experimental/distribution_strategy/resnet_estimator.py) example with TPUStrategy.

## Using `tf.distribute.Strategy` with custom training loops
As you've seen, using `tf.distribute.Strategy` with high level APIs is only a couple lines of code change. With a little more effort, `tf.distribute.Strategy` can also be used by other users who are not using these frameworks.

TensorFlow is used for a wide variety of use cases and some users (such as researchers) require more flexibility and control over their training loops. This makes it hard for them to use the high level frameworks such as Estimator or Keras. For instance, someone using a GAN may want to take a different number of generator or discriminator steps each round. Similarly, the high level frameworks are not very suitable for Reinforcement Learning training. So these users will usually write their own training loops.

For these users, we provide a core set of methods through the `tf.distribute.Strategy` classes. Using these may require minor restructuring of the code initially, but once that is done, the user should be able to switch between GPUs / TPUs / multiple machines by just changing the strategy instance.

Here we will show a brief snippet illustrating this use case for a simple training example using the same Keras model as before.
Note: These APIs are still experimental and we are improving them to make them more user friendly in TensorFlow 2.0.

First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.
"""

with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  optimizer = tf.keras.optimizers.SGD()

"""Next, we create the input dataset and call `tf.distribute.Strategy.experimental_distribute_dataset` to distribute the dataset based on the strategy."""

with mirrored_strategy.scope():
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(
    global_batch_size)
  dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

"""Then, we define one step of the training. We will use `tf.GradientTape` to compute gradients and optimizer to apply those gradients to update our model's variables. To distribute this training step, we put in in a function `step_fn` and pass it to `tf.distrbute.Strategy.experimental_run_v2` along with the dataset inputs that we get from `dist_dataset` created before:"""


@tf.function
def train_step(dist_inputs):
  def step_fn(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
      logits = model(features)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
      loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    return loss

  per_replica_losses = mirrored_strategy.experimental_run_v2(
    step_fn, args=(dist_inputs,))
  mean_loss = mirrored_strategy.reduce(
    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
  return mean_loss


"""A few other things to note in the code above:

1. We used `tf.nn.softmax_cross_entropy_with_logits` to compute the loss. And then we scaled the total loss by the global batch size. This is important because all the replicas are training in sync and number of examples in each step of training is the global batch. So the loss needs to be divided by the global batch size and not by the replica (local) batch size.
2. We used the `tf.distribute.Strategy.reduce` API to aggregate the results returned by `tf.distribute.Strategy.experimental_run_v2`. `tf.distribute.Strategy.experimental_run_v2` returns results from each local replica in the strategy, and there are multiple ways to consume this result. You can `reduce` them to get an aggregated value. You can also do `tf.distribute.Strategy.experimental_local_results` to get the list of values contained in the result, one per local replica.
3. When `apply_gradients` is called within a distribution strategy scope, its behavior is modified. Specifically, before applying gradients on each parallel instance during synchronous training, it performs a sum-over-all-replicas of the gradients.

Finally, once we have defined the training step, we can iterate over `dist_dataset` and run the training in a loop:
"""

with mirrored_strategy.scope():
  for inputs in dist_dataset:
    print(train_step(inputs))

"""In the example above, we iterated over the `dist_dataset` to provide input to your training. We also provide the  `tf.distribute.Strategy.make_experimental_numpy_dataset` to support numpy inputs. You can use this API to create a dataset before calling `tf.distribute.Strategy.experimental_distribute_dataset`.

Another way of iterating over your data is to explicitly use iterators. You may want to do this when you want to run for a given number of steps as opposed to iterating over the entire dataset.
The above iteration would now be modified to first create an iterator and then explicity call `next` on it to get the input data.
"""

with mirrored_strategy.scope():
  iterator = iter(dist_dataset)
  for _ in range(10):
    print(train_step(next(iterator)))

"""This covers the simplest case of using `tf.distribute.Strategy` API to distribute custom training loops. We are in the process of improving these APIs. Since this use case requires more work on the part of the user, we will be publishing a separate detailed guide in the future.

### What's supported now?
In TF 2.0 alpha release, we support training with custom training loops using `MirroredStrategy` as shown above. Support for other strategies will be coming in soon.
If you wish to use the other strategies like `TPUStrategy` in TF 2.0 with a custom training loop, you can currently do so by disabling eager execution (`tf.compat.v1.disable_eager_execution()`).  The code will remain similar, except you will need to use TF 1.x graph and sessions to run the training.
`MultiWorkerMirorredStrategy` support will be coming in the future.

### Examples and Tutorials
Here are some examples for using distribution strategy with custom training loops:

1. [Tutorial](../tutorials/distribute/training_loops.ipynb) to train MNIST using `MirroredStrategy`.
2. [DenseNet](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/densenet/distributed_train.py) example using `MirroredStrategy`.

## Other topics
In this section, we will cover some topics that are relevant to multiple use cases.

<a id="TF_CONFIG">
### Setting up TF\_CONFIG environment variable
</a>
For multi-worker training, as mentioned before, you need to set "TF\_CONFIG" environment variable for each
binary running in your cluster. The "TF\_CONFIG" environment variable is a JSON string which specifies what
tasks constitute a cluster, their addresses and each task's role in the cluster. We provide a Kubernetes template in the
[tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) repo which sets
"TF\_CONFIG" for your training tasks.

One example of "TF\_CONFIG" is:
```
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
})
```

This "TF\_CONFIG" specifies that there are three workers and two ps tasks in the
cluster along with their hosts and ports. The "task" part specifies that the
role of the current task in the cluster, worker 1 (the second worker). Valid roles in a cluster is
"chief", "worker", "ps" and "evaluator". There should be no "ps" job except when using `tf.distribute.experimental.ParameterServerStrategy`.

## What's next?

`tf.distribute.Strategy` is actively under development. We welcome you to try it out and provide and your feedback via [issues on GitHub](https://github.com/tensorflow/tensorflow/issues/new).
"""
