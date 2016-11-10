---
layout: post
title:  "结构化的全连接神经网络FNN框架搭建（TensorFlow）"
date:   2016-11-09 22:15:50 +0200
categories: machine_learning posts
---

上一篇中的全连接神经网络上是不具有代码重用性的，这样零散的结构不太符合object-oriented规范。今天照着tutorial里给的代码重写了一遍，理解了其中的一些，还有一些依旧难以理解，可能是对python语法也不太熟悉的缘故。

先把自己理解了的写下来，即使花费宝贵的半小时睡眠时间也得写下来，今日事今日毕！

[网络框架完整代码][]

[网络框架完整代码]: https://github.com/ShengleiH/machine_learning/tree/master/tensorflow/tutorials/encapsulatedFNN／naiveFNN.py
 

神经网络框架的搭建在naiveFNN.py中完成，这里面没有任何数据，相当于定义了一个巨型函数，所有的数据都在后续使用中填充，而naiveFNN.py单纯地构建一个**框架**（或称为函数）

> 首先强调一点：TensorFlow中，图片的所有输入和输出shape都是[batch\_size, NUM\_nuerons]；输入[55000, 784]，输出[55000, 10]

#### inference 推理函数--用来构建出网络雏形（在tensorflow中把这个网络结构称为graph，图）：

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

#### loss 损失函数--用来向雏形图中添加“损失操作”

```
def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss
```

**来看下这个函数的参数（args）：**

logits：是我们在inference中构建好的网络雏形，或者说也是网络最后输出层的结果。我觉得这两种理解都行，更偏向于第一种理解，个人感觉这个和matlab中CNN搭建时候很像--建立一个雏形网络，要向里面加东西的时候，就把这个雏形网络作为参数传进去。

labels：图片数据集的标签集，```shape = [batch_size, NUM_CLASSES]```。官网上说，这个集合必须是one-hot value，也就是说，如果这张图片中的数字是3，那么它的标签就得是[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]。这里把labels传进来肯定是非常必要的，因为你要根据这个正确的labels来判断你的网络好不好，然后back propagation来调整你的weights和biases呀！

**我的额外理解：**

1. ```loss = tf.reduce_mean(...)```，其实直接```loss = cross_entropy```也是没有啥关系的。只不过后面的learning rate调整得小一些就行啦，为什么呢？我是这样理解的：learning rate是每次梯度下降的步长，**learning rate越小**，梯度下降越慢，是一点儿一点，不敢懈怠地在下降，生动地说，就是**学习得相当仔细**。而我们的终极目标就是要减小loss到一个范围内，如果取均值，那么loss一开始就比较小了，于是learning rate可以相对大一些，也就是说学习得稍微粗略一些，也能在规定的时间（即迭代次数）内把loss减小到范围内；如果不取均值，那么loss一开始是比较大的，如果要在规定时间（即迭代次数）内把loss下降到一定的范围内，就要学习得仔细一点儿，即让learning rate相对小一些。个人理解，应该不太准确。
2. 这里用了```tf.nn.sparse_softmax_cross_entropy_with_logits(...)```，用其他的也是可以的，比如说上一篇中用的```tf.nn.softmax_cross_entropy_with_logits(...)```。

**我的问题**

我看了半天也没有看出来官网给的代码中的labels怎么就突然成了one-hot value了，因为它的代码中并没有如下转换代码，而且数据本身也不是one-hot value啊。

tutorials里面给了这段代码，可以让labels都变成one-hot value：

```
batch_size = tf.size(labels)
labels = tf.expand_dims(labels, 1)
indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
concated = tf.concat(1, [indices, labels])
onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
```

对tensorflow的tensor还不是很熟悉的我，真的很难想象tensor的dimension，我现在的唯一理解就是，第一维度就是最外层括号内有多少个整体（第二层括号或者数字），第二维度就是第二层括号内有几个整体（第三层括号或者数字），比如说[[],[],[],[]]的第一维度就是4，[[1,2,3],[3,2,1],[1,1,1],[2,2,2]]的第一维度是4，第二维度就是3。

所以用代码把每一个步骤的结果打印出来。

首先，假设```labels = [0, 3, 4, 1, 9]```，那么转换过程如下：

```
import tensorflow as tf
session = tf.Session()

NUM_CLASSES = 10

labels = [0, 3, 4, 1, 9]
print(labels)
#输出结果：[0, 3, 4, 1, 9]

batch_size = tf.size(labels)
print(session.run(batch_size))
#输出结果：5

labels = tf.expand_dims(labels, 1)
print(session.run(labels))
#输出结果：
[[0]
 [3]
 [4]
 [1]
 [9]]

range = tf.range(0, batch_size, 1)
print(session.run(range))
#输出结果：[0 1 2 3 4]

indices = tf.expand_dims(range, 1)
print(session.run(indices))
#输出结果：
[[0]
 [1]
 [2]
 [3]
 [4]]

concated = tf.concat(1, [indices, labels])
print(session.run(concated))
#输出结果：
[[0 0]
 [1 3]
 [2 4]
 [3 1]
 [4 9]]

pack = tf.pack([batch_size, NUM_CLASSES])
print(session.run(pack))
#输出结果：
[ 5 10]

onehot_labels = tf.sparse_to_dense(concated, pack, 1.0, 0.0)
print(session.run(onehot_labels))
#输出结果：
[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]

```

#### training 训练函数--用来向网络中（图中）添加“梯度下降”操作，故名“训练”

```
def training(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
```

**来看下这个函数的参数（args）：**

loss：添加了“损失函数”的网络，或者是这个网络的损失值。

learning_rate：学习率。说了这个函数是用梯度下降来训练网络的，所以学习率是必须的，也是人为规定的，所以作为参数输入。

**我的额外理解：**

1. 这里影响网络的是后面三行，第一行不影响，只是为了接下来的可视化做了一个监测。
2. optimizer.minimize更新了loss网络中的weights和biases，还更新了global_step（当前是第几次梯度下降）

**我的问题**

看不懂tf.scalar_summary的用法，说是这个可以把每一次的结果都实时地添加，然后就可以用TensorBoard来可视化整个过程了。还没仔细看过。

#### evaluation 评估函数--用来向**inference**函数构造的雏形网络中（图中）添加“评估函数”

```
def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
```

**来看下这个函数的参数（args）：**

logits：是我们在inference中构建好的网络雏形，或者说也是网络最后输出层的结果。

labels：图片数据集的标签集，是one-hot value，```shape = [batch_size, NUM_CLASSES]```

**我的额外理解：**

1. ```tf.nn.in_top_k(predictions, targets, k)```：predictions是```type = float32; shape = [batch_size, NUM_CLASSES]```的矩阵，其中的值代表这个图像被判定为该类别的概率，targets是```type = int32 or int64; shape = [batch_size]```的向量，其中的值代表这个图像的真实类别index。k表示这个图像的真实类别的值在predictions的概率中排名前k个。如果排到了前k个，则true，否则false。这里由于predictions全是one-hot value的，一个图像在10个类别下只有一个为1.0，其余都为0.0，所以k=1才能够分类（想一下如果k=2的话，概率的前两名就是1.0和0.0，不管这个类别是什么，它的概率肯定是1.0或者0.0的，也就是说，不论类别是啥，都会是对的。）
2. in_top_k函数的返回值是bool类型的，我们先把它转换成int类型的，也就是非1即0这样的，然后加一下总和，就知道有几个1了，也就是有几个正确的结果了。
3. redeuce_sum(...)的reduce意味着在这个tensor的第几个维度上操作。比如说，一个矩阵（Tensor's rank = 2），指定```dim = 0```，则在第一维上求和，比如:

```
x = [[2, 3, 1], [2, 3, 1]]

tf.reduce\_sum(x, 0)的结果就是：x = [[2, 3, 1]+[2, 3, 1]] = [4, 6, 2]
 
tf.reduce\_sum(x, 1)的结果就是：x = [[2+3+1], [2+3+1]] = [6, 6]
 
tf.reduce\_sum(x, -1) = tf.reduce\_sum(x, 0)
 
tf.reduce\_sum(x, -2) = tf.reduce\_sum(x, 1)
```

**我的问题**

这个没有啥问题......

> 下一篇中我们将填充数据，也就是官网代码中的fully_connected_feed.py中的代码






