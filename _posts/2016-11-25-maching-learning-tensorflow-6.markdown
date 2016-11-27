---
layout: post
title:  "TensorFlow图片数据读取和reshape"
date:   2016-11-25 18:35:49 +0800
categories: machine_learning posts
---

[cnn中数据读取代码][]

[cnn中数据读取代码]: https://github.com/ShengleiH/machine_learning/blob/master/tensorflow/tutorials/cifar10/cifar10_input.py

### 我认为最confusing的地方
 
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

**参考文档：**

[shuffle_batch运行原理][]

### 数据读取和reshape详解

接下来，一点儿点儿来分解和数据读取有关的代码。它们分布在cifar10\_input.py和cifar10\_model.py中。

cifar10\_model.py是路径指定和数据下载，cifar10\_input.py是数据读取和封装。

**下载CIFAR-10数据**

方法一：手动下载，直接去[http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz][]上下载这个压缩包。在工程目录下新建一个cifar10_data文件夹，把压缩包放到这个文件夹下。

方法二：通过url远程下载

这就需要用到python中的urllib包下的urlretrieve函数了 - **urllib.request.urlretrieve(...)**

```
API: 
urllib.request.urlretrieve(url, filepath=None, reporthook=None, data=None)

args:
url - 远程访问的地址
filepath - 保存到本地的路径，如果不指定，将自动生成一个临时路径
reporthook - 自定义的回调函数，当连接上服务器，而且相应的data传输完成时，将触发这个回调函数。
data - 要post到服务器的数据

return：
filepath: 如果没有filepath，那么将把临时路径返回，否则，就是那个指定路径
header：服务器端发过来的响应头

举个例子：
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
data_dir = 'cifar10_data'
filename = DATA_URL.split('/')[-1]
filepath = os.path.join(data_dir, filename)
filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=None, data=None)
```

> 官网中，给的data\_dir是‘/tmp/cifar10_data’，这里的'/tmp'是**系统**临时缓存路径，不是认为新建的，正常情况是隐藏的。

**解压CIFAR-10数据**

不管是手动下载还是url下载，刚下载下来的是压缩包，所以我们要先解压，然后才能使用。

方法一：手动解压，然后再把文件夹放到cifar10_data文件夹下。

方法二：python代码解压

需要用到python的tarfile包的open函数 - **tarfile.open(...).extractall(...)**

```
API:
tarfile.open(filepath, mode).extractall(dest_path)

args:
filepath - 需要解压的文件所在路径
mode - 解压的模式，如只读'r'，如读‘gz’后缀的压缩包‘r:gz’
dest_path - 解压到的路径

举个例子：
tarfile.open(filepath, ‘r:gz’).extractall(data_dir)
```

至此为止，不管用的是哪一种方法，都应该在你的工程下有如下目录结构：

![目录结构图]({{ site.baseurl }}/img/in-post/catelog_struc.png)

**数据读取**

把数存放到合适的路径之后，就可以开始读取数据，以及数据的后续处理了。

在cifar10\_input.py文件中定义一个```distorted_inputs(data_dir, batch_size)```函数来读取数据：

```
API: 
distorted_inputs(data_dir, batch_size)

args:
data_dir - 数据文件存放的路径，即‘cifar-10-batches-bin’文件夹
batch_size - 每次训练的图片数量

return：
images - batch_size个图片数据，shape为[batch_size, height, width, depth]
labels - batch_size个标签，shape为[batch_size]
```

函数中的具体步骤：

1) 解析路径

```
# 从该路径下的哪几个文件中获取数据
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1,6)]
# 监测这些文件是否存在
for f in filenames:
    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
# 生成一个先入先出的队列，文件阅读器会通过它来读取数据。
filename_queue = tf.train.string_input_producer(filenames)
```

2) 把这些文件中读出来的二进制内容封装成图片

自定义一个```read_cifar10```函数来进行图片封装操作，具体如何封装的等下说

```
read_input = read_cifar10(filename_queue)
reshaped_image = tf.cast(read_input.uint8image, tf.float32)
```

3) 为增加数据量，可以对图片进行“反转”、“平移”、“镜像”等操作

```
distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
distorted_image = tf.image.random_flip_left_right(distorted_image)
distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
float_image = tf.image.per_image_whitening(distorted_image)
```

4) 把读取出来的图片送入队列中，直到队列中有足够多的图片时，随机输出batch\_size个图片

```
return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)
```

自定义的```_generate_image_and_label_batch(...)```函数就是用来产生batch\_size个图像的，等下细说。

**图片封装 - read_cifar10**

文件中的数据是二进制的，我们先把这些数据包装成图片。

1) 声明一个一定大小的文件阅读器，大小为一张图画的大小。

```
 reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
```

2) 从文件中读出一张读片的数据，并转换类型，因为文件读取器读出来的是string类型的，我们需要转换成int类型的。

```
result.key, value = reader.read(filename_queue) 
record_bytes = tf.decode_raw(value, tf.uint8)
```

key - 该图片在文件中的index

value - 该图片数据，包括图片本身的pixels以及标签label

3) 把label和pixels分离开来

```
result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [result.depth, result.height, result.width])
```

这里可能会有疑惑，为什么是这样分离的呢？
我们来看一下cifar-10官方网站上的说明就很明显了：
 
> The binary version contains the files data\_batch\_1.bin, data\_batch\_2.bin, ..., data\_batch\_5.bin, as well as test_batch.bin. Each of these files is formatted as follows:
> 
> <1 x label><3072 x pixel>
> 
> ...
> 
> <1 x label><3072 x pixel>
> 
> In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image. 
 
所以，第一个数字就是label，后面的3072个数字，其实是这样组合的: 
 
[[red:1024], [green:1024], [blue:1024]] - 所以，reshape时候，第一维度是depth

在看其中一个色度中：

[red:1024] => [[row1:32], [row2:32], ..., [row32:32]] - 所以，第二维度是height，最后一个维度是width（即row）

所以reshape之后，图片的形状是：[depth, height, width]

4) 把图片本身转成[height, width, depth]形状

```
result.uint8image = tf.transpose(depth_major, [1, 2, 0])
```

**batch\_size个图像的生成 - \_generate\_image\_and\_label\_batch**

```
API: 
_generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle)

args:
image - 要送入队列的单张图片
label - 这张图片对应的标签
min_queue_examples - 队列中可容纳的数据量
batch_size - 要返回的图片数量
shuffle - 是否随机返回batch_size个图片

return：
images - batch_size个图片数据，shape为[batch_size, height, width, depth]
labels - batch_size个标签，shape为[batch_size]
```

这个其实就是调用了一下本文最开头讲的```tf.train.shuffle_batch(...)```这个函数，这个函数会把单张图片放进队列中，然后当队列中有足够的图片时，就随机返回batch\_size个图片。

如果shuffle为False，则调用```tf.train.batch(...)```这个函数，和```tf.train.shuffle_batch(...)```类似，只是返回batch\_size个图片的时候，不是随机的，而是顺序地输出。

OK，至此为止，核心代码都讲过了，剩下的都是重复的或者很简单的，就不多说了。

[shuffle_batch运行原理]: http://m.2cto.com/kf/201611/561584.html

[http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz]: http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz