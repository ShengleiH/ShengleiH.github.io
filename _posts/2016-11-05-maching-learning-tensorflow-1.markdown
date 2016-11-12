---
layout: post
title:  "使用TensorFlow搭建最简单的全连接神经网络"
date:   2016-11-05 11:49:45 +0800
categories: machine_learning posts
---

前几天看了TensorFlow，今天把之前写的“三层全连接神经网络”重新写了一遍，然后发现了一个问题，最后自己解决啦～  

[3层全连接网络完整代码][]

[3层全连接网络完整代码]: https://github.com/ShengleiH/machine_learning/blob/master/tensorflow/tutorials/naiveNN.py


#### 神经网络的搭建

###### 导入tensorflow包：
这要是不导入怎么用啊？！

```
import tensorflow as tf
```

###### 加载MNIST数据

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

###### 搭建空的网络框架

**1. 定义好将来存放数据x和标签y的占位符placeholder**

```
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

**2. 定义好用来初始化weights和bias的函数，然后定义好变量W和b**

```
def init_weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def init_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

W1 = init_weights([784, 100])
b1 = init_bias([100])
W2 = init_weights([100, 10])
b2 = init_bias([10])
```

**3. 前向传播**

使用relu函数作为激活函数，使用softmax作为分类器。

```
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)
```

**4. 反向传播**

tensorflow已经封装好了back propagation。
由于分类器使用了softmax，所以这里的代价函数是**交叉熵（cross-entropy）**

```
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

**5. 模型评估**

```
correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

###### 训练模型

**1. 开启事务session，初始化刚才定义的变量**

```
session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())
```

**2. 赋值训练**

```
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

###### 测试模型

```
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

#### 我遇到的问题（解决了）

我发现的一个问题。

每次在学习神经网络的时候，最搞不清楚的就是矩阵的维度，到底是nm？还是mn？这次在TensorFlow的代码中，也着实让我混乱了一把，因为它和我们理论学习中矩阵的维度是相反的！！！

在这里我们搭建了一个3层的全连接神经网络：第一层输入层有784个nueron（即一张图片为28*28像素的）；第二层隐藏层有100个nueron（自己设置的，无所谓吧，应该）；第三层输出层有10个neuron（即标签的数量0--9，10个数字）

于是在理论学习中，从输入层到隐藏层，我们会把**输入图像矩阵**设置为**784乘m**；**权重矩阵**设置成**100乘784**。

然而我们可以在代码中看到，TensorFlow是这样设置的：

```
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W1 = init_weights([784, 100])
```

也就是说，TensorFlow中把**输入图像矩阵**设置为**m乘784**；**权重矩阵**设置成**784乘100**,明显的，它的维度和我们理论学习中的是**相反的**！！！


那么我们在看**模型评估**这里：

```
correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))
```

argmax函数是把指定维度中最大的那一个的index返回。

前面代码中定义的placeholders中y\_的shape是[None, 10]，也就是说y\_的第一维度在TensorFlow中表示数据的index，第二维度才是计算出来的10个label的值呀。而在模型评估中，用了tf.argmax(y\_, 1)，就是说对y_的第一维度取最大值？！这不就错了？不应该使用第二维度吗？即tf.argmax(y\_, 2)？

然后我就试了一下tf.argmax(y\_, 2)，报错了！！！说“这个维度的取值在[0,2)”，于是，我明白了！TensorFlow中维度是从0开始算的！！！**很好，这相当程序员！！！**
