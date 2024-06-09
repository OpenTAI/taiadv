## Introduction
This is a white-box attack toolkit framework implemented in PyTorch, used to test the robustness of deep neural networks.

**Advantages:**
- More accurate robustness results can be obtained;
- A comprehensive summary of white-box attack methods;
- We obtain relatively accurate evaluation results for million-scale datasets.

**Innovative Contributions of Our Method:**
- "Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness"

We identify a new type of effect called imbalanced gradients, which can cause overestimated adversarial robustness and cannot be detected by detection methods for obfuscated gradients. And we propose Margin Decomposition (MD) attacks to exploit imbalanced gradients.

- "Probability Margin Attack: A Stronger Baseline for White-box Adversarial Robustness Evaluation"

We propose a novel attack method called Probability Margin Attack (PMA) that introduces a probability margin loss(PM).


## Related Work
We have summarized the commonly used loss functions and attack strategies in white-box attacks.
- Loss Function List

<img src="loss.jpg" width="400" height="150">

Let $z$ be the logit output of $f(x)$ and $z_i$ be the logits output of the $i$-th class, and $p_i$ be the probability output of the $i$-th class for a total number of $N$ classes. Sorting the values of $z_i$ in ascending order, $z_{\pi_i}$ represents the $i$-th largest logit value (except $z_{y}$).
$z_{max}$ / $p_{max}$ is the maximum value of $z_i$ / $p_i$ for $i &ne; y$, respctively.

- Attack Method List
  
| Strategy | Loss Function | Remarks |
|:--------|:-------------|:-------|
| PGD | ce/cet/dlr/mg/pm/mi/alt | Single attack, untargeted attack |
| APGD | ce/cet/dlr/mg/pm/mi | Single attack, untargeted attack |
| APGDT | dlr/mg/pm | Single-target attack, multi-objective attack |
| MD | mg | Single-target attack, untargeted attack |
| FAB | - | Single-target attack, untargeted attack |
| PMA | pm | Single-target attack, untargeted attack |
| PMA+ | - | Ensemble attacks, PMA+APGDT |


## Usage

### Installation
```bash
git clone https://github.com/taiadv/taiadv/attacks/White-box.git
```

### Single-GPU operation
```bash
python main.py --dataset <dataset_name> --datapath <dataset_dir> --model <model_path> --eps 8 --bs <batchsize> --attack_type <PMA> --loss_f <pm> --num_steps 100 --num_classes <num_classes>
```
### Multi-GPU parallelism
```bash
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> main.py --dataset <dataset_name> --datapath <dataset_dir> --model <model_path> --eps 8 --bs <batchsize> --attack_type <PMA> --loss_f <pm> --num_steps 100 --num_classes <num_classes>
```

### Citation
```bash
@article{madry2017towards,
  title={Towards deep learning models resistant to adversarial attacks},
  author={Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  journal={arXiv preprint arXiv:1706.06083},
  year={2017}
}
@inproceedings{tashiro2020diversity,
  title={Diversity can be transferred: Output diversification for white-and black-box attacks},
  author={Tashiro, Yusuke and Song, Yang and Ermon, Stefano},
  booktitle={NeurIPS},
  year={2020}
}
@inproceedings{croce2020minimally,
  title={Minimally distorted adversarial examples with a fast adaptive boundary attack},
  author={Croce, Francesco and Hein, Matthias},
  booktitle={ICML},
  year={2020},
}
@inproceedings{carlini2017towards,
  title={Towards evaluating the robustness of neural networks},
  author={Carlini, Nicholas and Wagner, David},
  booktitle={IEEES&P},
  year={2017}
}
@inproceedings{yu2023efficient,
  title={Efficient loss function by minimizing the detrimental effect of floating-point errors on gradient-based attacks},
  author={Yu, Yunrui and Xu, Cheng-Zhong},
  booktitle={CVPR},
  year={2023}
}
@inproceedings{croce2020reliable,
  title={Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},
  author={Croce, Francesco and Hein, Matthias},
  booktitle={ICML},
  year={2020}
}
@article{antoniou2022alternating,
  title={Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks},
  author={Antoniou, Nikolaos and Georgiou, Efthymios and Potamianos, Alexandros},
  journal={arXiv preprint arXiv:2212.07992},
  year={2022}
}
```


