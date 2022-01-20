# 生成对抗网络（GAN）的前沿进展（论文、报告、框架和Github资源）汇总



2014年，Ian Goodfellow等人在《GenerativeAdversarial Nets》一文中首次提出了GANs，标志着GANs的诞生。



原文链接：https://arxiv.org/pdf/1406.2661v1.pdf

PPT链接：http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf

源码链接：https://github.com/goodfeli/adversarial

视频链接：https://www.youtube.com/watch?v=HN9NRhm9waY





本文总结了一系列关于GANs的前沿工作进展

 

一、最新研究论文（根据Google Scholar的引用数进行降序排列）

 

1. 基于深度卷积生成对抗网络的无监督学习（Unsupervised Representation Learning with Deep   Convolutional Generative Adversarial Networks (DCGANs)）2015

   原文链接：https://arxiv.org/pdf/1511.06434v2.pdf

2. 对抗实例的解释和利用（Explaining and Harnessing   Adversarial Examples）2014

   原文链接：https://arxiv.org/pdf/1412.6572.pdf

3. 基于深度生成模型的半监督学习（ Semi-Supervised Learning with Deep   Generative Models ）2014

   原文链接：https://arxiv.org/pdf/1406.5298v2.pdf

4. 基于拉普拉斯金字塔生成式对抗网络的深度图像生成模型（Deep Generative Image Models using a Laplacian Pyramid   of Adversarial Networks）2015

   原文链接：http://papers.nips.cc/paper/5773-deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks.pdf

5. 训练GANs的一些技巧（Improved Techniques for Training GANs）2016

   原文链接：https://arxiv.org/pdf/1606.03498v1.pdf

6. 条件生成对抗网络（Conditional Generative   Adversarial Nets）2014

   原文链接：https://arxiv.org/pdf/1411.1784v1.pdf

7. 生成式矩匹配网络（Generative Moment Matching Networks）2015

   原文链接：http://proceedings.mlr.press/v37/li15.pdf

8. 超越均方误差的深度多尺度视频预测（Deep multi-scale video   prediction beyond mean square error）2015

   原文链接：https://arxiv.org/pdf/1511.05440.pdf

9. 通过学习相似性度量的超像素自编码（Autoencoding beyond pixels using a learned similarity metric）2015

   原文链接：https://arxiv.org/pdf/1512.09300.pdf

10. 对抗自编码（Adversarial   Autoencoders）2015

    原文链接：https://arxiv.org/pdf/1511.05644.pdf

11. InfoGAN:基于信息最大化GANs的可解释表达学习（InfoGAN:Interpretable Representation Learning by Information Maximizing Generative   Adversarial Nets）2016

    原文链接：https://arxiv.org/pdf/1606.03657v1.pdf

12. 上下文像素编码：通过修复进行特征学习（Context   Encoders: Feature Learning by Inpainting）2016

    原文链接：http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf

13. 生成对抗网络实现文本合成图像（Generative   Adversarial Text to Image Synthesis）2016

    原文链接：http://proceedings.mlr.press/v48/reed16.pdf

14. 基于像素卷积神经网络的条件生成图片（Conditional  Image Generation with PixelCNN Decoders）2015

    原文链接：https://arxiv.org/pdf/1606.05328.pdf

15. 对抗特征学习（Adversarial Feature   Learning）2016

    原文链接：https://arxiv.org/pdf/1605.09782.pdf

16. 结合逆自回归流的变分推理（Improving Variational   Inference with Inverse Autoregressive Flow ）2016

    原文链接：https://papers.nips.cc/paper/6581-improving-variational-autoencoders-with-inverse-autoregressive-flow.pdf

17. 深度学习系统对抗样本黑盒攻击（Practical   Black-Box Attacks against Deep Learning Systems using Adversarial Examples）2016

    原文链接：https://arxiv.org/pdf/1602.02697.pdf

18. 参加，推断，重复：基于生成模型的快速场景理解（Attend,   infer, repeat: Fast scene understanding with generative models）2016

    原文链接：https://arxiv.org/pdf/1603.08575.pdf

19. f-GAN: 使用变分散度最小化训练生成神经采样器（f-GAN: Training Generative Neural Samplers using Variational Divergence   Minimization ）2016

    原文链接：http://papers.nips.cc/paper/6066-tagger-deep-unsupervised-perceptual-grouping.pdf

20. 在自然图像流形上的生成视觉操作（Generative   Visual Manipulation on the Natural Image Manifold）2016

    原文链接：https://arxiv.org/pdf/1609.03552.pdf

21. 通过平均差异最大优化训练生成神经网络（Training generative neural networks via Maximum Mean Discrepancy optimization）2015

    原文链接：https://arxiv.org/pdf/1505.03906.pdf

22. 对抗性推断学习（Adversarially   Learned Inference）2016

    原文链接：https://arxiv.org/pdf/1606.00704.pdf

23. 基于循环对抗网络的图像生成（Generating images with recurrent adversarial networks）2016

    原文链接：https://arxiv.org/pdf/1602.05110.pdf

24. 生成对抗模仿学习(Generative Adversarial Imitation Learning）2016

    原文链接：http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf

25. 基于3D生成对抗模型学习物体形状的概率隐空间（Learning a Probabilistic Latent Space of Object Shapes   via 3D Generative-Adversarial Modeling）2016

    原文链接：https://arxiv.org/pdf/1610.07584.pdf

26. 学习画画（Learning What   and Where to Draw）2016

    原文链接：https://arxiv.org/pdf/1610.02454v1.pdf

27. 基于辅助分类器GANs的条件图像合成（Conditional Image   Synthesis with Auxiliary Classifier GANs）2016

    原文链接：https://arxiv.org/pdf/1610.09585.pdf

28. 隐生成模型的学习（Learning in Implicit   Generative Models）2016

    原文：https://arxiv.org/pdf/1610.03483.pdf

29. VIME: 变分信息最大化探索（VIME: Variational   Information Maximizing Exploration）2016

    原文链接：http://papers.nips.cc/paper/6591-vime-variational-information-maximizing-exploration.pdf

30. 生成对抗网络的展开（Unrolled   Generative Adversarial Networks）2016

    原文链接：https://arxiv.org/pdf/1611.02163.pdf

31. 训练生成对抗网络的基本方法（Towards Principled Methods for Training Generative Adversarial Networks）2017

    原文链接：https://arxiv.org/pdf/1701.04862.pdf

32. 基于内省对抗网络的神经图像编辑（Neural Photo Editing with Introspective Adversarial  Networks）2016

    原文链接：https://arxiv.org/pdf/1609.07093.pdf

33. 基于解码器的生成模型的定量分析（On the  Quantitative Analysis of Decoder-Based Generative Models ）2016

    原文链接：https://arxiv.org/pdf/1611.04273.pdf

34. 结合生成对抗网络和Actor-Critic 方法（Connecting Generative   Adversarial Networks and Actor-Critic Methods）2016

    原文链接：https://arxiv.org/pdf/1610.01945.pdf

35.  通过对抗网络使用模拟和非监督图像训练（ Learning   from Simulated and Unsupervised Images through Adversarial Training）2016

    原文链接：https://arxiv.org/pdf/1612.07828.pdf

36. 基于上下文RNN-GANs的抽象推理图的生成（Contextual   RNN-GANs for Abstract Reasoning Diagram Generation）2016

    原文链接：https://arxiv.org/pdf/1609.09444.pdf

37. 生成多对抗网络（Generative   Multi-Adversarial Networks）2016

    原文链接：https://arxiv.org/pdf/1611.01673.pdf

38. 生成对抗网络组合（Ensembles of  Generative Adversarial Network）2016

    原文链接：https://arxiv.org/pdf/1612.00991.pdf

39. 改进生成器目标的GANs(Improved generator objectives for GANs) 2016

    原文链接：https://arxiv.org/pdf/1612.02780.pdf

40. 生成对抗模型的隐向量精准修复（Precise   Recovery of Latent Vectors from Generative Adversarial Networks）2017

    原文链接：https://openreview[.NET](http://lib.csdn.net/base/dotnet)/pdf?id=HJC88BzFl

41. 生成混合模型（Generative   Mixture of Networks）2017

    原文链接：https://arxiv.org/pdf/1702.03307.pdf

42. 记忆生成时空模型（Generative Temporal   Models with Memory）2017

    原文链接：https://arxiv.org/pdf/1702.04649.pdf

43. 停止GAN暴力：生成性非对抗模型（Stopping GAN Violence: Generative Unadversarial   Networks）2017

    原文链接：https://arxiv.org/pdf/1703.02528.pdf

 

二、理论学习

1. 训练GANs的技巧，

参见链接：http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans.pdf

2. Energy-Based GANs 以及Yann Le Cun 的相关研究

参见链接：http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

3. 模式正则化GAN

参见链接：https://arxiv.org/pdf/1612.02136.pdf

 

三、报告

1. Ian Goodfellow的GANs报告

参见链接：http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf  

2. Russ Salakhutdinov的深度生成模型

参见链接：http://www.cs.toronto.edu/~rsalakhu/talk_Montreal_2016_Salakhutdinov.pdf

 

 

四、课程/教程

1. NIPS 2016教程：生成对抗网络

参见链接：https://arxiv.org/pdf/1701.00160.pdf

2. 训练GANs的技巧和窍门

参见链接：https://github.com/soumith/ganhacks

3. OpenAI生成模型

参见链接：https://blog.openai.com/generative-models/

4. 用Keras实现MNIST生成对抗模型

参见链接：https://oshearesearch.com/index.[PHP](http://lib.csdn.net/base/php)/2016/07/01/mnist-generative-adversarial-model-in-keras/

5. 用[深度学习](http://lib.csdn.net/base/deeplearning)TensorFlow实现图像修复

参见链接：http://bamos.github.io/2016/08/09/deep-completion/

 

五、Github资源以及模型

1. 深度卷积生成对抗模型（DCGAN）

参见链接：https://github.com/Newmu/dcgan_code

2. TensorFlow实现深度卷积生成对抗模型（DCGAN）

参见链接：https://github.com/carpedm20/DCGAN-tensorflow

3. Torch实现深度卷积生成对抗模型（DCGAN）

参见链接：https://github.com/soumith/dcgan.torch

4. Keras实现深度卷积生成对抗模型（DCGAN）

参见链接：https://github.com/jacobgil/keras-dcgan

5. 使用神经网络生成自然图像（Facebook的Eyescream项目）

参见链接：https://github.com/facebook/eyescream

6. 对抗自编码（AdversarialAutoEncoder）

参见链接：https://github.com/musyoku/adversarial-autoencoder

7. 利用ThoughtVectors 实现文本到图像的合成

参见链接：https://github.com/paarthneekhara/text-to-image

8. 对抗样本生成器（Adversarialexample generator）

参见链接：https://github.com/e-lab/torch-toolbox/tree/master/Adversarial

9. 深度生成模型的半监督学习

参见链接：https://github.com/dpkingma/nips14-ssl

10. GANs的训练方法

参见链接：https://github.com/openai/improved-gan

11. 生成式矩匹配网络（Generative Moment Matching Networks, GMMNs）

参见链接：https://github.com/yujiali/gmmn

12. 对抗视频生成

参见链接：https://github.com/dyelax/Adversarial_Video_Generation

13. 基于条件对抗网络的图像到图像翻译（pix2pix）

参见链接：https://github.com/phillipi/pix2pix

14. 对抗[机器学习](http://lib.csdn.net/base/machinelearning)库Cleverhans,

参见链接：https://github.com/openai/cleverhans



五、框架以及学习库（根据GitHub的星级排序）

  1.谷歌的TensorFlow [C++ and CUDA]

主页链接：https://www.tensorflow.org/

Github链接：https://github.com/tensorflow/tensorflow

  2. Berkeley Vision and LearningCenter (BVLC) 的Caffe [C++]

主页链接：http://caffe.berkeleyvision.org/

Github链接：https://github.com/BVLC/caffe

安装指南：http://gkalliatakis.com/blog/Caffe_Installation/README.md

3. François Chollet的Keras [[Python](http://lib.csdn.net/base/python)]

主页链接：https://keras.io/

Github链接：https://github.com/fchollet/keras

  4. Microsoft Cognitive Toolkit -CNTK [C++]

主页链接：https://www.microsoft.com/en-us/research/product/cognitive-toolkit/

Github链接：https://github.com/Microsoft/CNTK

5. Amazon 的MXNet [C++]

主页链接：http://mxnet.io/

Github链接：https://github.com/dmlc/mxnet

  6. Collobert, Kavukcuoglu &Clement Farabet的Torch，被Facebook广泛采用[Lua]

主页链接：http://torch.ch/

Github链接：https://github.com/torch

1. Andrej Karpathy 的Convnetjs [[JavaScript](http://lib.csdn.net/base/javascript)]

   主页链接：http://cs.stanford.edu/people/karpathy/convnetjs/

   Github链接：https://github.com/karpathy/convnetjs

2. Université de Montréal的 Theano [[python](http://lib.csdn.net/base/python)]

   主页链接：http://deeplearning[.net](http://lib.csdn.net/base/dotnet)/software/theano/

   Github链接：https://github.com/Theano/Theano

3. startup Skymind 的Deeplearning4j [[Java](http://lib.csdn.net/base/java)]

   主页链接：https://deeplearning4j.org/

   Github链接：https://github.com/deeplearning4j/deeplearning4j

4. Baidu 的Paddle[C++]

   主页链接：http://www.paddlepaddle.org/

   Github链接：https://github.com/PaddlePaddle/Paddle

5. Amazon 的Deep Scalable Sparse Tensor Network Engine (DSSTNE) [C++]

   Github链接：https://github.com/amzn/amazon-dsstne   

6. Nervana Systems 的Neon [Python & Sass]

   主页链接：http://neon.nervanasys.com/docs/latest/

   Github链接：https://github.com/NervanaSystems/neon

7. Chainer [Python]

   主页链接：http://chainer.org/

   Github链接：https://github.com/pfnet/chainer

8. h2o [Java]

   主页链接：https://www.h2o.ai/

   Github链接：https://github.com/h2oai/h2o-3

9. Istituto Dalle Molle di   Studi sull’Intelligenza Artificiale (IDSIA) 的Brainstorm [Python]

   Github链接：https://github.com/IDSIA/brainstorm

10. Andrea Vedaldi 的Matconvnet by [Matlab]

    主页链接：http://www.vlfeat.org/matconvnet/

    Github链接：https://github.com/vlfeat/matconvnet

更多细节请参考原文链接：http://gkalliatakis.com/blog/delving-deep-into-gans