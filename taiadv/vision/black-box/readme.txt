# 简介
本项目是一个迁移对抗攻击视觉模型的开源工具箱。支持基于多个预训练特征提取器进行对抗样本的生成和评测。同时提供了预先生成好的对抗数据集，可以直接下载后进行评测，无需再生成。

## 方法原理
![load image failed](./image/overview.png "对抗样本生成方法示意图")

我们借助多个预训练的图像编码器，在特征层并行扰动，生成高度可迁移的对抗样本。

## 亮点
### 多卡多模，增强迁移
我们借助多卡多模技术来增强对抗样本迁移性。我们使用8个主流预训练图片特征提取器进行特征提取，在特征层进行扰动生成高度可迁移的对抗样本。

### 百万评测，通用准确
我们提供了基于cc3m构造的百万级别数据集cc1m，通过大量样本的评测更准确的评估模型的鲁棒性。（通用性）
一个数据集可评测下游所有任务。

### 开箱即用，无需生成
我们提供已经预先生成好的多个对抗版本数据集，包括ImagNet–1k，COCO2017，ADE20k，cc1m等，开箱即用，无需用户再次生成。

# 教程
## 安装
通过github下载源码进行使用。

## 使用
### 对抗样本数据集生成
#### 参数列表
| 参数名 | 类型 | 含义 |
| --- | --- | --- |
| dataset       | string | 数据集名称 |
| data_path     | string | 数据集根路径 |
| batch_size    | int | 批次大小 |
| random        | bool | 随机初始化 |
| loss          | string | 损失函数 |
| epsilon       | float | 攻击预算 |
| num_steps     | int | 攻击步数 |
| step_size     | float | 步长 |
| optim         | string | 优化对抗噪声的优化器 |
| jpeg_resistant| bool  | 是否启用抵御jpg压缩的对抗攻击 |

#### 运行
```
python attack.py 
```

### 模型迁移对抗性评估
#### 示例：用于图像分类任务鲁棒性评估
```
python eval/imagenet_classification.py --model_name=vit_base_patch16_224 --model_path=<your/model/path> --output=./ --clean_path=<your/imagenet/path> --adv_path=<your/adversarial imagenet/path>
```

# 致谢
感谢timm库提供的模型架构和权重，以及其他开源数据集。以下详细介绍我们用到的特征提取器和数据集。

## 特征提取器
我们选用了timm库提供的8个主流的特征提取器用于生成对抗样本，其中包含了多种不同的模型架构和预训练方法，具体如下：
| 模型名 | 原论文 |
| --- | --- |
| vgg16 | Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014. |
| resnet101 | He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778. |
| efficient net | Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
| convnext_base | Liu Z, Mao H, Wu C Y, et al. A convnet for the 2020s[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11976-11986. |
| vit_base_patch16_224 | Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020. |
| vit_base_patch16_224.dino | Caron M, Touvron H, Misra I, et al. Emerging properties in self-supervised vision transformers[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 9650-9660. |
| beit_base_patch16_224 | Bao H, Dong L, Piao S, et al. Beit: Bert pre-training of image transformers[J]. arXiv preprint arXiv:2106.08254, 2021. |
| swin_base_patch4_window7_224 | Liu Z, Lin Y, Cao Y, et al. Swin transformer: Hierarchical vision transformer using shifted windows[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 10012-10022. |

## 数据集

| 数据集名 | 原论文 |
| --- | --- |
| CC3M | Sharma P, Ding N, Goodman S, et al. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018: 2556-2565. |
| ImageNet | Deng J, Dong W, Socher R, et al. Imagenet: A large-scale hierarchical image database[C]//2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009: 248-255. |
| COCO | Lin T Y, Maire M, Belongie S, et al. Microsoft coco: Common objects in context[C]//Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer International Publishing, 2014: 740-755. |
| ADE20k | Zhou B, Zhao H, Puig X, et al. Scene parsing through ade20k dataset[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 633-641. |
| CheXpert | Irvin J, Rajpurkar P, Ko M, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison[C]//Proceedings of the AAAI conference on artificial intelligence. 2019, 33(01): 590-597. |
