# 简介
本项目是一个白盒攻击视觉模型的开源工具箱，支持现阶段多个白盒攻击方法的评测。

## 亮点
#### 增加了更强的单体攻击方法PMA，更高效的组合攻击方法PMA+
我们提出了新的攻击方法PMA（Probability Margin Attack），使用了新提出的损失函数PM（Probability Margin Loss），并提出了更高效的组合攻击方法PMA+


#### 对抗攻击方法比较全面，支持更灵活的使用
- 在单攻击评测中，支持不同策略和损失函数的组合；
- 多攻击评测中，支持不同攻击方法的组合。

|攻击方法|论文名称|
|----|----|
|PGD|“Towards deep learning models resistant to adversarial attacks”|
|ODI|“Diversity can be transferred: Output diversification for white-and black-box attacks”|
|PGD_mg|“Towards evaluating the robustness of neural networks”|
|MT|“An alternative surrogate loss for pgd-based adversarial testing”|
|APGD|“Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks”|
|APGDT|“Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks”|
|FAB|“Minimally distorted adversarial examples with a fast adaptive boundary attack”|
|MD|“Imbalanced gradients: a subtle cause of overestimated adversarial robustness”|
|PGD_alt|“Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks”|
|PGD_mi|“Efficient loss function by minimizing the detrimental effect of floating-point errors on gradient-based attacks”|

#### 支持更多的数据集，还支持百万级别鲁棒性评估
我们提供了CIFAR10，CIFAR100，ImageNet的数据处理和评测，除此之外，我们提供了基于cc3m构造的百万级别数据集cc1m，通过百万样本的评测得到更准确的模型鲁棒性。

# 教程
## 安装
```bash
https://github.com/fra31/auto-attack.git
```

## 使用

### 参数设置

|参数名称|参数含义|
|----|----|
|dataset|数据集名称（CIFAR10/CIFAR100/ImageNet/CC1M）|
|datapath|数据集路径|
|modelpath|模型路径|
|eps|扰动范围（一般取4或8）|
|bs|批量大小|
|attack_type|攻击策略|
|random_start|是否加入随机噪声（bool类型）|
|num_restarts|重启的次数|
|num_steps|攻击步数|
|loss_f|损失函数的类型|
|use_odi|是否使用ODI策略|
|num_classes|模型分类数量|
|result_path|结果保存的地方|


### 运行
```bash
python main.py --dataset <dataset_name> --datapath <dataset_dir> --model <model_path> --eps 8 --bs <batchsize> --attack_type <PMA> --loss_f <pm> --num_steps 100 --num_classes <num_classes>
```



