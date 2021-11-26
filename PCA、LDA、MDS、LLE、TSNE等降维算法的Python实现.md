**PCA****、****LDA****、****MDS****、****LLE****、****TSNE****等降维算法的****Python****实现** 

2019-10-14 14:58

整理 | 夕颜

【导读】网上关于各种降维算法的资料参差不齐，但大部分不提供源代码。近日，有人在 GitHub 上整理了一些经典降维算法的 Demo(Python)集合，同时给出了参考资料的链接。

\1. **PCA**

​                               

资料链接：https://blog.csdn.net/u013719780/article/details/78352262

https://blog.csdn.net/u013719780/article/details/78352262

https://blog.csdn.net/weixin_40604987/article/details/79632888

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/PCA

\1. **KPCA**

 

资料链接：https://blog.csdn.net/u013719780/article/details/78352262

https://blog.csdn.net/weixin_40604987/article/details/79632888

https://blog.csdn.net/u013719780/article/details/78352262

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/PCA

\1. **LDA**

 

资料链接：https://blog.csdn.net/ChenVast/article/details/79227945

https://www.cnblogs.com/pinard/p/6244265.html

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/LDA

\1. **MDS**

 

 

资料链接：https://blog.csdn.net/zhangweiguo_717/article/details/69663452?locationNum=10&fps=1

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/MDS

\1. **ISOMAP**

 

资料链接：https://blog.csdn.net/zhangweiguo_717/article/details/69802312

http://www-clmc.usc.edu/publications/T/tenenbaum-Science2000.pdf

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/ISOMAP

\1. **LLE**

 

资料链接：https://blog.csdn.net/scott198510/article/details/76099630

https://www.cnblogs.com/pinard/p/6266408.html?utm_source=itdadao&utm_medium=referral

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/LLE

\1. **TSNE**

 

资料链接：http://bindog.github.io/blog/2018/07/31/t-sne-tips/

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/T-SNE

\1. **AutoEncoder**

 

\1. **FastICA**

资料链接：https://blog.csdn.net/lizhe_dashuju/article/details/50263339

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/ICA

\1. **SVD**

资料链接：https://blog.csdn.net/m0_37870649/article/details/80547167

https://www.cnblogs.com/pinard/p/6251584.html

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/SVD

\1. **LE**

 

资料链接：https://blog.csdn.net/hustlx/article/details/50850342

https://blog.csdn.net/jwh_bupt/article/details/8945083

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/LE

\1. **LPP**

 

资料链接：https://blog.csdn.net/qq_39187538/article/details/90402961

https://blog.csdn.net/xiaohen123456/article/details/82288222

GitHub代码：https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/master/codes/LPP

此外，作者还指出本次整理的降维算法实现环境为 Python3.6、ubuntu18.04(windows10) ，需要的库包括 numpy、sklearn、tensorflow 和 matplotlib，且具有以下特点：

·     每一个代码都可以单独运行，但是只是作为一个demo，仅供学习使用；

·     其中 AutoEncoder 只是使用 AutoEncoder 简单地实现了一个 PCA 降维算法，自编码器涉及到了深度学习领域，其本身就是一个非常大的领域；

·     LE 算法的鲁棒性极差，对近邻的选择和数据分布十分敏感；

·     2019.6.20 添加了 LPP 算法，但是效果没有论文上那么好，有点迷，后续需要修改。

项目 GitHub 链接：https://github.com/heucoder/dimensionality_reduction_alo_codes

（*本文为 AI科技大本营投稿文章， 转 载请微 信联系 1092722531 ）

 