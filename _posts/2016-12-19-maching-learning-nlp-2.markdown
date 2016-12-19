---
layout: post
title:  "NLP Coursera By Michael Collins - Week2"
date:   2016-12-19 12:20:49 +0800
categories: machine_learning posts
---

[课程下载和PPT下载（密码：i72q）][]

第二周的课程讲了TAGGING问题，即如何给句子tag上词态，如下：

![tagging example]({{ site.baseurl }}/img/in-post/tagging_examples.png)

也就是说，如果输入：The dag laughs，就需要模型输出：DT NN VB

整个过程如下所示：

![tagging structure]({{ site.baseurl }}/img/in-post/overall_structure_for_tagging.png)

现在将这个过程转换成概率论模型：

![hmm1]({{ site.baseurl }}/img/in-post/hmm1.jpg)

![hmm2]({{ site.baseurl }}/img/in-post/hmm2.jpg)

![hmm3]({{ site.baseurl }}/img/in-post/hmm3.jpg)

![hmm4]({{ site.baseurl }}/img/in-post/hmm4.jpg)

关于隐马尔可夫的详解，可以参考[知乎上的回答][]。

[课程下载和PPT下载（密码：i72q）]: https://pan.baidu.com/s/1dFrAcjJ

[知乎上的回答]:https://www.zhihu.com/question/20962240
