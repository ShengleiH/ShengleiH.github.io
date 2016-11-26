---
layout: post
title:  "TensorFlow图片数据读取和reshape"
date:   2016-11-25 18:35:49 +0800
categories: machine_learning posts
---

[cnn中数据读取代码][]

[cnn中数据读取代码]: https://github.com/ShengleiH/machine_learning/blob/master/tensorflow/tutorials/cnn／cifar10_input.py
 
为什么tf.train.shuffle_batch()函数的参数中，只输入了单张图片，但是在训练的时候却能够获取到一个图片batch呢？

在代码中可以看到是这样用的：

```
images, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples
)
```

很神奇地，输入的是[image, label]--单张图片，输出的却是images, label_batch--一个数据集。

其实只是我们看上去是放入了一张图片，其实tensorflow对这个函数是这样理解的：shuffle\_batch构建了一个RandomShuffleQueue，并不断地把单个的[image，label]对送入队列中，这个入队操作是通过QueueRunners启动另外的线程来完成的。这个RandomShuffleQueue会顺序地压样例到队列中，直到队列中的样例个数达到了batch\_size+min\_after\_dequeue个。它然后从队列中选择batch\_size个随机的元素进行返回。

也就是说，有一个线程在训练开始后，不断地在调用read_cifar10()这个函数，从文件中不断地读取图片，送入队列，直到数量够为止。

[shuffle_batch运行原理][]

[shuffle_batch运行原理]: http://m.2cto.com/kf/201611/561584.html