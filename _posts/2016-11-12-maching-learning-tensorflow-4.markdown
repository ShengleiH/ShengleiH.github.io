---
layout: post
title:  "IRIS flowers识别（TensorFlow.contrib.learn）"
date:   2016-11-12 14:10:49 +0800
categories: machine_learning posts
---

官方教程的第四篇，识别IRIS flowers

[iris代码][]

[iris代码]: https://github.com/ShengleiH/tensorflow_tutorials/blob/master/tensorflow/tutorials/iris_flowers

使用了tensorflow.contrib.learn中已经分装好的神经网络。

**包的导入**

```
import tensorflow as tf
import numpy as np
```
import numpy：接下来有用到numpy中的数据类型。

**下载数据**

把iris flowers的csv文件下载下来，这里把文件放在和神经网络同一文件夹下，所以路径直接是文件名称就可以了，不然的话就需要调整一下。


```
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING, features_dtype=np.float, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST, features_dtype=np.float, target_dtype=np.int)
```

> 这里需要注意，官方的tutorials中有错误，tensorflow更新了，但是教程没有及时更新。
> 官方tutorials中给的代码中使用的是```load_csv()```，但是在目前最新版本的tensorflow中已经把load\_csv()给移除了，取而代之的是```load_csv_with_header(filename,
                         target_dtype,
                         features_dtype,
                         target_column=-1)```和```load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            target_column=-1)```

> github上有这两个函数的[源代码][]。

[源代码]: https://github.com/tensorflow/tensorflow/blob/fcab4308002412e38c1a6d5c6145119f04540d45/tensorflow/contrib/learn/python/learn/datasets/base.py#L38

**配置网络**

tf.contrib.learn.DNNClassifier中已经把网络的框架封装完毕，只需要我们传入“特征配置”、“每一层节点数量”、“目标分类数”和”checkpoint data存放的路径“即可。

特征配置：设定特征的数据类型-这里符合“real_valued_column”，然后一共有4个特征，所以dimension=4

```
# 特征配置
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
# 配置网络
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir="/tmp/iris_model")
```

**训练网络**

直接调用fit函数即可，参数为训练数据、其对应的labels以及需要迭代的次数

```
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)
```

**评估网络**

直接调用evaluate函数即可，参数为测试数据。

```
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
```

最终结果为：Accuracy: 0.966667

**预测数据**

用numpy.array模拟两个数据，然后调用predict函数预测模拟数据的labels

```
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))
```
结果为：Predictions: [1 2]

参考资料：

[TensorFlow官方tutorial-iris][]

[StackOverFlow-load_csv][]

[TensorFlow官方tutorial-iris]: https://www.tensorflow.org/versions/r0.11/tutorials/tflearn/index.html

[StackOverFlow-load_csv]: http://stackoverflow.com/questions/40007785/why-tensor-flow-could-not-load-csv
