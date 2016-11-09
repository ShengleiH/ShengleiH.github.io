---
layout: post
title:  "使用TensorFlow搭建结构化的全连接神经网络"
date:   2016-11-09 22:15:50 +0200
categories: jekyll update
---

上一篇中的全连接神经网络上是不具有代码重用性的，这样零散的结构不太符合object-oriented规范。今天照着tutorial里给的代码重写了一遍，理解了其中的一些，还有一些依旧难以理解，可能是对python语法也不太熟悉的缘故。

先把自己理解了的写下来，即使花费宝贵的半小时睡眠时间也得写下来，今日事今日毕！

[封装的全连接网络完整代码][]

[封装的全连接网络完整代码]: https://github.com/ShengleiH/machine_learning/tree/master/tensorflow/tutorials/encapsulatedFNN


#### 神经网络框架搭建

这一步在naiveFNN.py中完成，这里面没有任何数据，所有的数据都在fully_connected_feed.py中填充，而naiveFNN.py单纯地构建一个**框架**

> 首先强调一点：TensorFlow中，图片的所有输入和输出shape都是[batch\_size, NUM\_nuerons]；输入[55000, 784]，输出[55000, 10]

###### inference 推理函数--用来构建出网络雏形（在tensorflow中把这个网络结构称为graph，图）：

这里我们构建一个4层的神经网络，输入层[784个nuerons]、隐藏层1（hidden1）、隐藏层2（hidden2）和输出层（softmax_linear）

```
def weights_varialble(shape, input_units):
    initial = tf.truncated_normal(shape, stddev=1.0/math.sqrt(float(input_units)))
    return tf.Variable(initial, name='weights')


def biases_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name='biases')
    
    
def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = weights_varialble([IMAGE_PIXELS, hidden1_units], IMAGE_PIXELS)
        biases = biases_variable([hidden1_units])
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = weights_varialble([hidden1_units, hidden2_units], hidden1_units)
        biases = biases_variable([hidden2_units])
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear'):
        weights = weights_varialble([hidden2_units, NUM_CLASSES], hidden2_units)
        biases = biases_variable([NUM_CLASSES])
        logits = tf.matmul(hidden2, weights) + biases

    return logits
```

**来看下这个函数的参数（args）：**

images：输入的图片集--训练集、测试集......形状（shape）为[batch\_size, IMAGE\_PIXELS]，如[50, 784]

hidden1\_units, hidden2\_units：这两层的neuron数量

**我的额外理解：**

1. with tf.name_scope('hidden1')：就是在名为hidden1的范围下定义了下面的这些东西：weights、biases和hidden1，所有的这些东西都是属于hidden1的。
2. 这里weights和biases的初始化的函数不一定是这样的，如biases的初始化还可以用上一篇中的：```tf.constant(0.1, shape=shape)```
3. 还需要注意的，tensorflow中，matmul函数中参数的顺序，理论学习中我们知道应该是```y = Wx + b```，而这里matmul中的顺序是相反的--```matmul(x, W)```。其实我发现，tensorflow中矩阵的行列和理论学习中的都是相反的......
4. 这里输出层softmax\_linear没有使用调用softmax函数，是因为tensorflow中有这样的函数可以一边对数据apply softmax，一边进行计算交叉熵cross_entropy（简写xentropy），我们在下一步的loss函数中会用到的--sparse\_softmax\_cross\_entropy\_with\_logits(...)

**我的问题**

name_scope中定义的变量可以在这个范围之外被读取到吗？如果说不能，那么上面代码中，```return logits```是怎么来的？如果说能，那么上面代码中，hidden1和hidden2中都有weights和biases这可怎么区分啊？？？

###### loss 损失函数--用来向雏形图中添加“损失操作”

```
def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss
```

----
明天继续更......











