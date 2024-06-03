# PMA
"Probability Margin Attack: A Stronger Baseline for White-box Adversarial Robustness Evaluation"

We propose a novel attack method called Probability Margin Attack (PMA) that introduces a probability margin loss.


- Loss Function List


| Adversarial Attack Loss | Form of Loss Function |
|:-----------------------|:---------------------|
| Untargeted CE | |
| Targeted CE | |
| DLR | |
| Margin | |
| Probability Margin | |

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
git clone https://github.com/yourusername/whiteboxattack.git
```

### Example



