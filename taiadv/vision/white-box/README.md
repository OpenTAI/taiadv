# Introduction
The project is an open-source toolbox for white-box attacks on visual models, supporting the evaluation of multiple white-box attack methods currently available.
# Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks

## Highlights
#### Enhanced Single Attack Method PMA and More Efficient Combined Attack Method PMA+
We introduce a new attack method called PMA (Probability Margin Attack), which utilizes the newly proposed Probability Margin Loss function. Additionally, we present a more efficient combined attack method, PMA+.
📄 **[Paper (arXiv 2411.15210)](https://arxiv.org/abs/2411.15210)**

We propose a novel adversarial attack method, **Probability Margin Attack (PMA)**, based on the newly introduced **Probability Margin Loss**. Furthermore, we present **PMA+**, an enhanced version that combines diverse strategies for improved efficiency and performance.

#### Comprehensive and Flexible Adversarial Attack Methods
- For single attack evaluations, our toolbox supports combinations of different strategies and loss functions.
- In multi-attack evaluations, it allows for the combination of various attack methods.
To support large-scale, reproducible robustness evaluations, we provide a complete toolkit for attacking models on **CIFAR-10**, **CIFAR-100**, **ImageNet**, and our newly released **CC1M** dataset—a curated 1M-sample benchmark derived from CC3M.

---

## 🔧 Installation

#### Expanded Dataset Support and Large-Scale Robustness Evaluation
We provide data processing and evaluation tools for CIFAR10, CIFAR100, and ImageNet. Furthermore, we introduce a million-sample dataset, cc1m, constructed from cc3m, enabling more accurate model robustness assessments through large-scale evaluation.

# Tutorial
## Installation
```bash
https://github.com/fra31/auto-attack.git
```

## Usage

### Parameter Settings

| Parameter Name | Type |Description |
| ---- | ---- | ----|
| dataset | string  | Name of the dataset (CIFAR10/CIFAR100/ImageNet/CC1M) |
| datapath | string | Path to the dataset |
| modelpath | int | Path to the model |
| eps | int | Perturbation range (commonly set to 4 or 8) |
| bs | int  | Batch size |
| attack_type | string | Attack strategy |
| random_start | bool | Whether to add random noise (boolean) |
| num_restarts | int | Number of restarts |
| num_steps | int | Number of attack steps |
| loss_f | string | Type of loss function |
| use_odi | bool | Whether to use the ODI strategy |
| num_classes | int | Number of classes in the model |
| result_path | string | Path to save the results |

### Execution
```bash
python main.py --dataset <dataset_name> --datapath <dataset_dir> --model <model_path> --eps 8 --bs <batchsize> --attack_type <PMA> --loss_f <pm> --num_steps 100 --num_classes <num_classes>
```


# Acknowledgements
We have integrated several classic white-box attack methods, incorporating various strategies, as detailed below:

|Methods|Paper Title|
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

git clone https://github.com/fra31/auto-attack.git
cd auto-attack
pip install -r requirements.txt
