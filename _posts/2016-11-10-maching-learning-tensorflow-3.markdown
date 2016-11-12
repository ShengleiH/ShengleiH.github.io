---
layout: post
title:  "结构化的全连接神经网络FNN框架--part2使用（TensorFlow）"
date:   2016-11-10 14:10:49 +0800
categories: machine_learning posts
---

这一篇中我们将使用前面搭建的FNN网络来训练MNIST数据集

[使用FNN代码][]

[使用FNN代码]: https://github.com/ShengleiH/machine_learning/blob/master/tensorflow/tutorials/encapsulatedFNN/fully_connected_feed.py
 
这里不进行整篇代码的详细解释，只对其中几个我疑惑的地方进行解释。

#### 几个包的导入

```
from __future__ import division

import os.path
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import naiveFNN

```
from \_\_future\_\_ import division：为了使用'//'，整除，结果为不大于真是结果的整数。如果不引入这个包，就会默认为'/'就是除法。

import os.path：为了使用```checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')```这是用来记录每一步的日志，和网络本身没啥关系。

import time：为了使用```start_time = time.time()```这是用来获取迭代时间的，和网络本身没有关系

import tensorflow as tf：为了使用tensorflow中的方法

from tensorflow.examples.tutorials.mnist import input_data：从官网github上下载实验所需数据集，如果自行下载了，和代码放在同一目录下，然后直接import input\_data就好了。

import naiveFNN：使用FNN网络框架。这里要注意，如果要运行代码，请务必将naiveFNN.py下载，并且和这个代码放在同一目录下。

#### 网络各层参数定义

```
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
FLAGS = flags.FLAGS
```

这里的FLAGS，我的理解就是和java中的枚举类似，之后调用的时候就直接```FLAGS.balabala```

**我的额外理解**

一开始我感觉很奇怪，为什么train\_dir的值会是data，前面的实验中我们都使用的是MNIST\_data，难道这可以随便设吗？

后来我去翻了下Dataset.read_data\_sets(...)方法的源代码（已经在你电脑中的python下面了哦）:


```
local_file = base.maybe_download(TRAIN_IMAGES, train_dir, SOURCE_URL + TRAIN_IMAGES)
```
 
继续去base.maybe\_download(...)源代码查看：

```
def maybe_download(filename, work_directory, source_url):
```

可见，这只是一个work directory，也就是在本地，你要把下载下来的数据存放的地方，如果这个directory不存在，就创建一个，如果存在，就放在这个下面。真正决定从哪里下载的是source_url。filename也只是这些数据集在本地的命名。

#### 在run_training函数中给框架填充数据

```
logits = naiveFNN.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
loss = naiveFNN.loss(logits, labels_placeholder)
train_op = naiveFNN.training(loss, FLAGS.learning_rate)
eval_correct = naiveFNN.evaluation(logits, labels_placeholder)
```

#### 在run_training函数中训练网络

```
for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
```

迭代FLAGS.max\_steps次，每次去batch_size个数据。

**我的额外理解**

fill\_feed\_dict：自定义的一个用来向placehoder中填充数据的函数，从data\_sets.train中取出数据，填充到images\_placeholder, labels\_placeholder中，然后返回一个包含了这两个placeholders的dictionary。

sess.run([train\_op, loss], feed\_dict=feed\_dict)：写成```sess.run([train_op, loss], feed_dict)```也可以的，有没有```feed_dict=```这个无所谓。这个函数是前面第一个参数是什么，它就返回什么，比如说，第一个参数是[train\_op, loss]，那么就会返回[train\_op, loss]，然后因为train\_op的值我们不需要，所以就用'_'来忽略，loss的结果就用loss\_value来接收。

xrange：作用和range是一样的，但是在大数据集时，xrange的性能更好。

```
range(5)
[0, 1, 2, 3, 4]

xrange(5)
list(xrange(5))
[0, 1, 2, 3, 4]
```

**最后整理一下run_training的整个步骤**

1. 从train set中取batch_size个数据并填充到网络中
2. 运行网络，获取网络的loss值
3. 继续步骤1、2，直到循环“max_steps”次（只是网络的累积调整过程，每一次都是在上一次的基础上进行的）
4. 每100次循环输出一个loss value看看
5. 每999次循环以及**最后一次循环**，对当前网络进行一个测评：
   1. 从train／validation／test数据集中取batch_size个数据
   2. 运行测评网络（naiveFNN.evaluation()那个函数），获取这batch_size个数据中正确的个数
   3. 继续步骤1、2，直到循环“数据集总个数//batch_size”次（这里‘//’是一个整除）。循环的目的是为了尽可能地用尽数据集中的数据。
   4. 最后获取到train／validation／test数据集中判断正确的个数
   5. 计算得到准确率```precision = true_count / num_examples```。由于整除这里的num_examples可能比实际的少```num_examples = real_num_examples // batch_size * FLAGS.batch_size```
6. 于是在**最后一次循环**后，我们获取到了一个准确率precision

OK，以上就是用TensorFlow搭建封装的神经网络的解析。很多API还是没有理解好，不过不急，慢慢地用多了就有感觉了，就会知道了。

参考资料：[TensorFlow官方tutorial-mechanics-101][]

[TensorFlow官方tutorial-mechanics-101]: https://www.tensorflow.org/versions/r0.11/tutorials/mnist/tf/index.html#tensorflow-mechanics-101






