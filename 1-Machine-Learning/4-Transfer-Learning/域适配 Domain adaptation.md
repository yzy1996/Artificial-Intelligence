# 域适配 Domain adaptation

**分类任务**



首先分为源域数据（source domain）和目标域数据（target domain）



源域数据是有标签的，目标域数据是没有标签的



现在想通过源域数据训练一个分类器，迁移应用到目标域数据



为什么能实现呢？



关键在于将这两个数据的特征空间统一，就能用同一个分类器分类



缺点是：这两个源的图片是很相似的，比如都是手写数字





论文-DANN Domain-Adversarial Training of Neural Networks 2016