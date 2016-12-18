---
layout: post
title:  "NLP Coursera By Michael Collins - Week1"
date:   2016-12-18 14:20:49 +0800
categories: machine_learning posts
---

[课程下载和PPT下载（密码：i72q）][]

前段时间一直在纠结到底应该看那个公开课来入门NLP，希望能够找到一个像Andrew Ng的machine learning一样的课程。先看了CS224d，打击了自信心，看到lecture3就没再看下去了。现在只能看得进Cousera那种一节一节的小课了。

Finally, 在ybb的推荐的下，找到了这个课，奈何Coursera已经不提供这个课程了，好在有百度网盘神一般的存在！所以开始看这个课程。

Collins的语言很精练，口齿特别清楚，没有字幕都没有任何问题（网盘里面的字幕用不了的）。全程干货（毕竟Coursera的课时比较短，所以废话比较少），但是想理解的更深入，需要看Collins的[notes]，每周对应一个note。

Collins最好的一点是，他的符号很清楚，不会出现那种对不上的现象，很好理解，然后证明过程什么的简直明了，一点儿都不藏着掖着！

今天看完了week1的课程，总结一下这一节课的内容。

### 构建模型框架 - Markov Process

毕竟是机器学习嘛，所以第一步，先要把实际问题转化成数学模型。在NLP中，一般使用的都是概率模型，即把语言模型变成概率论范畴。

比如说，现在有一段语音，说的很含糊，没有听清楚，好像是“like your”，又好像是“lie cured”。那么到底是哪一种呢？我们就看在现有的语料库中，到底是“like your”出现的概率大，还是“lie cured”的概率大。

于是就把语音识别问题转变成了一个概率问题：输入一串字符，输出这串字符组合在一起的概率，如果概率大，就是正确的句子。下面构建这个模型

![马尔可夫过程]({{ site.baseurl }}/img/in-post/markov.jpg)

至此，模型框架搭建完毕，但是参数还没有设定好。也就是说，现在如果向模型中随便输入一个句子，要求输出的结果是这个句子出现的概率。那么我们就需要事先知道模型中，每一个p(w\|u,v)。用如下方式来计算：

![概率计算]({{ site.baseurl }}/img/in-post/probability_calculation.jpg)

对于上面的计算方法，不能解决出现概率为0，但实际这句句子是合理的情况。下面介绍两种方法，来对上面的计算方法进行改进：

第一种是Linear Interpolation：

![Linear Interpolation]({{ site.baseurl }}/img/in-post/linear_interpolation.jpg)

这里的三个系数用下面的方法进行选择：

![parameter selection]({{ site.baseurl }}/img/in-post/selection_parameters.jpg)

其实这个方法就是将三种计算概率的方式**线性结合**起来，具体的结合方式有很多种，上面只是其中一种。

上面这个方法中的三个系数和概率本身没有关系，但是更好的方法是让他们有关系：

bucketing法：对不同范围内的counts，使用不同的系数

![bucketing]({{ site.baseurl }}/img/in-post/bucketing.PNG)

将三个系数都写成同一个参数的线性组合：

![linear method]({{ site.baseurl }}/img/in-post/linear_method.png)

第二种是Discounting Method:

这个方式就是，从概率不为0的情况中分出一部分的概率给概率为0的情况。

![discounting method]({{ site.baseurl }}/img/in-post/discounting_method.PNG)

至此为止，整个模型搭建完毕。

### 评价模型

一般情况下：

![perplexity]({{ site.baseurl }}/img/in-post/perplexity.PNG)

当服从均匀分布的时候：

![perplexity special case]({{ site.baseurl }}/img/in-post/perplexity_special.PNG)

以上就是这一周的课程中讲的内容。notes中的内容和这个一样，基本没有什么补充。

下面用一张流程图来总结一下整个模型的构建过程。

![summary]({{ site.baseurl }}/img/in-post/summary_week1.jpg)


[课程下载和PPT下载（密码：i72q）]: https://pan.baidu.com/s/1dFrAcjJ

[notes]:http://www.cs.columbia.edu/~mcollins/notes-spring2013.html
