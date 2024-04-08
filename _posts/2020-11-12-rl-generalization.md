---
layout: post
title: "[调研] 如何提高强化学习算法模型泛化能力初探"
date: 2020-11-12 10:14
modified_date: 
author:
- 吴悦晨
excerpt: 在深度学习中，模型很容易过拟合到参与训练的数据集。因此，深度学习训练模型的时候通常会将数据集分成训练集和测试集，保证训练的模型在测试集上仍然有很好的性能，即模型的泛化能力。在深度强化学习的应用中，模型的泛化能力也同样重要。本文将介绍最近深度强化学习领域中提高模型泛化能力的一些方法，如域随机化、正则等。
categories: [ 强化学习 泛化性 ]
---
> 原发表链接: [【伏羲讲堂】如何提高强化学习算法模型泛化能力初探](https://zhuanlan.zhihu.com/p/328287119)

> 在深度学习中，模型很容易过拟合到参与训练的数据集。因此，深度学习训练模型的时候通常会将数据集分成训练集和测试集，保证训练的模型在测试集上仍然有很好的性能，即模型的泛化能力。在深度强化学习的应用中，模型的泛化能力也同样重要。本文将介绍最近深度强化学习领域中提高模型泛化能力的一些方法，如域随机化、正则等。

## 泛化

什么是泛化（generalization）呢？  

这里引用维基百科上关于泛化的定义：  

[A generalization is a form of abstraction whereby common properties of specific instances are formulated as general concepts or claims. Generalizations posit the existence of a domain or set of elements, as well as one or more common characteristics shared by those elements (thus creating a conceptual model). As such, they are the essential basis of all valid deductive inferences (particularly in logic, mathematics and science), where the process of verification is necessary to determine whether a generalization holds true for any given situation.](https://en.wikipedia.org/wiki/Generalization)  

在机器学习任务中，泛化能力被认为是在训练集上训练后，在“未见过”的数据集上的性能。当然，这里常常有一个假设，训练的数据集和测试的数据集是同分布的，即来自于同一个领域（domain）的同一个任务（task）。

## 领域和任务
这里引用万引迁移学习[1]中关于领域（domain）和任务（task）的定义：  

领域D={X, P(X)}由两个部分组成：特征空间X和特征空间的边缘分布P(X)。如果两个领域不同，它们的特征空间或边缘概率分布不同。  

A domain is a pair D={X, P(X)}, which consists of two components: a feature space X and a marginal probability distribution P(X). In general, if two domains are different, then they may have different feature spaces or different marginal probability distributions.  

任务T={Y, P(Y\|X)}组成：给定一个领域D={X, P(X)}的情况下，一个任务也包含两个部分：标签空间Y和一个目标预测函数f(.)。目标预测函数不能被直接观测，但可以通过训练样本学习得到。从概率论角度来看，目标预测函数f(.)可以表示为P(Y\|X)。  

A task is defined as a pair T={Y, P(Y\|X)}. Given a specific domain, D={X, P(X)}, a task consists of two components: a label space Y and an objective predictive function f(.), which is not observed but can be learned from the training data. From a probabilistic viewpoint, f(x) can be written as p(y\|x).

## 泛化和迁移

如下图所示，本文讨论的是来自同一个领域同一个任务的模型泛化能力，红色框中的部分。而那些领域不同或者任务不同的情况被认为是迁移学习的范畴，蓝色框中的部分。

![](/assets/images/blog/20201112/01.png)

对于强化学习来说，泛化能力指的是：对于同一个领域中的同一个任务，强化学习模型在训练环境训练，在测试环境验证的模型的性能。

## 强化学习的泛化

因为机器学习中的泛化能力一般是指同分布的不同数据集上的模型的性能。

这里，主要考虑两种情况来说明强化学习模型的泛化能力：在场景A（训练集）中训练，在场景B（测试集）中测试。 	
1. 场景A的状态集和场景B的状态集S不一样，状态转移矩阵P一样。比如对手风格A中训练，对手风格B中泛化测试。 	
2. 场景A的状态集和场景B的状态集S类似，状态转移矩阵P不一样。比如游戏版本A中训练，游戏版本B中泛化测试。

强化学习的数据是有模型与环境交互过程中产生的，所以普遍认为强化学习中的泛化能力弱，模型较容易过拟合到当前的训练环境。目前比较常见的提高强化学习模型泛化能力的方式主要有两种：Regularization和Randomization。 	 	

![](/assets/images/blog/20201112/02.png)

## Regularization-CoinRun

这里，首先介绍一篇2019年发表在ICML上的论文，Quantifying generalization in reinforcement learning[2]。

首先，如图所示，这篇文章中开源了一个用来验证深度强化学习算法泛化性能的游戏环境——CoinRun。这个游戏环境拥有上百万不同的关卡，每个关卡都有相应的难度等级（从难度1-难度3），因此可以用来验证强化课程学习的算法性能。

![](/assets/images/blog/20201112/03.png)

文章中还进行了很多基准实验：

#### 对模型架构的评估

通过调整模型的构架，对比不同容量大小的模型的泛化能力。文章中采取了3中不同的模型架构：

Nature-CNN: PPO+3 CNNs

IMPALA-CNN: IMPALA+3 residual blocks

IMPALA-Large: IMPALA+5 residual blocks

如图所示，可以看出，模型容量越大，算法的泛化能力越强。

![](/assets/images/blog/20201112/04.png)

#### 对正则项的评估

通过调整模型正则项，对比不同的正则项对模型泛化能力的影响。文章中进行了5种不同的消融实验，baseline算法为IMPALA-CNN。从图中可以看出，正则项对模型的泛化能力都是有积极的作用的。

![](/assets/images/blog/20201112/05.png)

## Regularization-Information Bottleneck

文章Dynamics Generalization via Information Bottleneck in Deep Reinforcement Learning[3]认为，深度强化学习模型可以分为两个部分，感知层和决策层。感知层的作用是将环境观测转换为状态表示，而决策层负责从状态表示中学习控制策略。但是感知层比较容易过拟合到当前的训练环境，文章提出了一种信息瓶颈（Information Bottleneck），尽可能地限制从环境观测传递到状态标识的信息，鼓励神经网络去学习一些高维地特征，算法可以有效的泛化到未见过的环境。如下是文章中的主要框架图。

![](/assets/images/blog/20201112/06.png)

这是文章中主要用到的KL散度公式，和相应的正则和梯度计算方式。

![](/assets/images/blog/20201112/07.png)

文章中声明信息瓶颈比基线算法和其他正则化技术(如L-2正则和Dropout)拥有更好的泛化能力。

## Randomization-Domain Randomization

文章Domain randomization for transferring deep neural networks from simulation to the real world[4]提出了一种样本增强的新方式——域随机化。在一些难以直接采样的环境中，比如机器人的抓取任务，真实环境交互学习代价非常高。文章中提出利用域随机化的方式在仿真环境中进行大量实验可以很好的迁移到正式环境。

![](/assets/images/blog/20201112/08.png)

域随机化的方向有以下几个方面：

![](/assets/images/blog/20201112/09.png)

从实验结果的表格中可以看出：
* 训练的时候加入干扰物是至关重要的。 
* 随机相机位置也能提供一点帮助，但是对最终性能的影响不大。 
* 在预处理中加入随机噪声没什么效果，但是少量的噪声可以帮助收敛。

![](/assets/images/blog/20201112/10.png)

仿真环境实验的时候可以采用域随机化的方式，缓解神经网络过拟合到单一环境的问题。可以让特征提取网络不关注低维的环境动态参数，更关注提取环境中的高维特征。

## Randomization-Active Domain Randomization

文章Active domain randomization[5]在域随机化的基础上提出了，主动的域随机化技术。提出用强化学习去学环境参数的调整方向，用判别器预测奖励（reward），使域随机化的更有效率。

![](/assets/images/blog/20201112/11.png)

## Randomization-Network Randomization

文章Network randomization: A simple technique for generalization in deep reinforcement learning[6]提出了一种十分有效的样本增强方式，可以一次采样，多次训练。通过提高模型泛化能力的同时，又提高了样本的利用率。

![](/assets/images/blog/20201112/12.png)

这是文章中网络随机化主要用到的公式：

![](/assets/images/blog/20201112/13.png)

文章中进行了一组有趣的对比试验，模型在黑猫白狗的训练集中训练，在白猫黑狗的测试集中测试。模型很容易过拟合到训练样本，提取了一些无关紧要的特征：颜色。通过网络随机化这种样本增强方式，可以增强网络提取特征的能力。同时，相比于传统的图像样本增强方式：1）cutout (CO)；2）grayout (GR)；3）inversion (IV)；4）color jitter (CJ)，文章中的网络随机化的样本增强方式更有效率。

![](/assets/images/blog/20201112/14.png)

如下图所示是文章进行大量消融实验的结果，可以看出在不同的算法中，网络随机化保证了最优的模型泛化能力。

![](/assets/images/blog/20201112/15.png)

参考文献：  
[1] Pan S J, Yang Q. A survey on transfer learning[J]. IEEE Transactions on knowledge and data engineering, 2009, 22(10): 1345-1359.  
[2]Cobbe K, Klimov O, Hesse C, et al. Quantifying generalization in reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2019: 1282-1289.  
[3]Lu X, Lee K, Abbeel P, et al. Dynamics Generalization via Information Bottleneck in Deep Reinforcement Learning[J]. arXiv preprint arXiv:2008.00614, 2020.  
[4]Tobin J, Fong R, Ray A, et al. Domain randomization for transferring deep neural networks from simulation to the real world[C]//2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017: 23-30.  
[5]Mehta B, Diaz M, Golemo F, et al. Active domain randomization[C]//Conference on Robot Learning. PMLR, 2020: 1162-1176.  
[6]Lee K, Lee K, Shin J, et al. Network randomization: A simple technique for generalization in deep reinforcement learning[J]. arXiv, 2019: arXiv: 1910.05396.
