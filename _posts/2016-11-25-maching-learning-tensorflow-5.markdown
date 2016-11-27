---
layout: post
title:  "TensorFlow卷积神经网络搭建（CIFAR10）"
date:   2016-11-25 17:34:49 +0800
categories: machine_learning posts
---

官方教程的CNN网络，识别CIFAR10数据集

[cnn代码][]

[cnn代码]: https://github.com/ShengleiH/machine_learning/blob/master/tensorflow/tutorials/cifar10/cifar10_model.py

这篇文章将使用“敏捷开发模型”来写......顾名思义，从最主要的部分“网络结构”开始写，然后自定义一些方法，如“数据的读取”，假设这些方法已经可用，最后再去补充这些方法。个人感觉，这样由主要到次要，慢慢补充才能捕获整个结构。最后的最后，再来看如何使用TensorBoard可视化整个过程，所以开始的代码中，省略了所有的summary。

### CNN网络搭建

**CNN网络结构**

输入的图像shape为 [batch\_size, height, width, depth]

第一层：卷积层 conv1

第二层：池化层 pool1

第三层：标准化层 norm1

第四层：卷积层 conv2

第五层：标准化层 norm2

第六层：池化层 pool2

第七层：全联接层 local1

第八层：全联接层 local2

第九层：输出层 softmax_linear

```
def inference(images_after_reshape):
    # convolution layer 1
    with tf.variable_scope('conv1') as conv1_scope:
        kernel = _variable_with_weight_decay('weights', shape=[5,5,3,64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(images_after_reshape, kernel, strides=[1,1,1,1], padding='SAME')
        biases = _variable_on_cpu('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias=biases)
        conv1 = tf.nn.relu(conv, name=conv1_scope.name)

    # pooling layer
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    # normalization
    norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # convolution layer 2
    with tf.variable_scope('conv2') as conv2_scope:
        kernel = _variable_with_weight_decay('weights', shape=[5,5,64,64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = _variable_on_cpu('biases', shape=[64], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.bias_add(conv, bias=biases)
        conv2 = tf.nn.relu(conv, name=conv2_scope.name)

    # normalization
    norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')

    # pooling layer
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')

    # fully connected layer 1
    with tf.variable_scope('local1') as local1_scope:
        images_as_long_vector = tf.reshape(pool2, shape=[FLAGS.batch_size, -1])
        input_neurons = images_as_long_vector.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[input_neurons, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', shape=[384], initializer=tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(images_as_long_vector, weights) + biases, name=local1_scope.name)

    # fully connected layer 2
    with tf.variable_scope('local2') as local2_scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', shape=[192], initializer=tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(local1, weights) + biases, name=local2_scope.name)

    # output layer - softmax layer
    with tf.variable_scope('softmax_linear') as softmax_scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local1, weights) + biases, name=softmax_scope.name)

    return softmax_linear
```

这里，有三个地方时我们自己假设的：

1. 假设输入图像的形状已经是[batch\_size, height, width, depth]。具体如何让自己的图像变成这样的形状，稍后会讲。

2. 假设已经定义了```_variable_with_weight_decay(name, shape], stddev, wd)```函数，该函数将返回含有惩罚项（如L2, L1 loss）的权重。

3. 假设已经定义了```_variable_on_cpu(name, shape, initializer)```函数，该函数将返回通过initializer获得的随机数。

**CNN网络loss计算**

为了调整权重，我们需要有一个loss function来评估网络的好坏。

```
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss
```

1. logits：就是CNN网络，即inference()函数的返回值。

2. ```tf.add_to_collection('losses', cross_entropy_mean)```：losses这个collection中除了有```cross_entropy_mean```，还有```weight_decay```这是在
定义```_variable_with_weight_decay```时加进去的。

3. ```tf.add_n```：就是把losses这个collection中的```cross_entropy_mean```和```weight_decay```取出来，然后相加，这样就形成了含有惩罚项(L2 or L1)的损失函数。

**CNN网络权重调整--梯度下降**

在定义了损失函数之后，我们将一边使用这个损失函数评估网络，一边使用梯度下降法(GD)来调整网络中的权重和偏置们。

梯度下降法的使用，分为两步--1.每一个权重的梯度gradient；2.根据learning rate，使用梯度下降法调整权重。

```
def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

    # Begin training
    optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op
```

这里由于我们不知道合适的learning rate是多少，所以我们动态地调整lr，画出曲线（如何画出曲线，稍后再说，这里把画曲线的代码省略了），然后最后选取最合适的lr。

### 开始补充自定义方法

**初始化权重**

```
def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)
    return var
```

这两个自定义函数中，```_variable_on_cpu```很好理解，就是根据提供的initializer来随机生成一些数。

```_variable_with_weight_decay```中主要是两个参数stddev和wd：

1. stddev：因为这个函数中用的initializer是“截断高斯分布”，需要提供一个“标准差”来确定这个分布。

2. wd：weight_decay。这是惩罚项系数，就是乘在L2, L1 loss前面的那个系数。如果wd=0.0，就意味着，不要惩罚项。

3. 可见，weights的初始化都用到了weight_dacay，而biases的初始化不需要weight_decay。

**图像数据的读取和reshape**

将在下一篇中讲到，因为比较复杂。

参考资料：

[TensorFlow官方tutorial-cnn-github][]

[TensorFlow官方tutorial-cnn-github]: https://github.com/tensorflow/tensorflow/tree/r0.11/tensorflow/models/image/cifar10