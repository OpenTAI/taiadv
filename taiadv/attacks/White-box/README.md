# WhiteBoxAttack
A framework for measuring model robustness through white-box attacks

## List
- [Introduction](#Introduction)
- [Attack Methods](#Attack-Methods)
- [Usage](#Usage)
- [References](#References)


## Introduction

WhiteBoxAttack is a framework designed for conducting white-box attacks on machine learning models, especially neural networks. By accessing the internal structure and parameters of the model, it generates adversarial examples to test the robustness of the model.

## Attack Methods

- Loss Function List
  
| Adversarial Attack Loss | Form of Loss Function |
|:-----------------------:|:---------------------:|
| Untargeted CE | |
| Targeted CE | |
| DLR | |
| Margin | |
| Probability Margin | |

- Attack Method List
  
| Strategy | Loss Function | Remarks |
|:--------:|:-------------:|:-------:|
| PGD | ce/cet/dlr/mg/pm/mi/alt | Single-target attack |
| APGD | ce/cet/dlr/mg/pm/mi | Single-target attack |
| APGDT | dlr/mg/pm | Single-target attack, multi-objective attack |
| MD | mg | Single-target attack |
| FAB | - | Single-target attack |
| PMA | pm | Single-target attack |
| PMA+ | - | Combined attack, PMA+APGDT |

## Usage

### Installation
```bash
git clone https://github.com/yourusername/whiteboxattack.git
```


