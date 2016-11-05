---
layout: post
title:  "使用TensorFlow搭建最简单的全链接神经网络"
date:   2016-11-5 17:15:45 +0200
categories: jekyll update
---
前几天看了TensorFlow，今天把之前写的“三层全连接神经网络”重新写了一遍，然后发现了一个问题，最后自己解决啦～

0. 导入tensorflow库，不导入你要怎么用？！

{% highlight tensorflow %}
import tensorflow as tf
{% endhighlight %}

1. 加载MNIST数据

{% highlight tensorflow %}
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
{% endhighlight %}