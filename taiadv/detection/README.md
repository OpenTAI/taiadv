## An open-source toolkit for Adversarial Example Detection (AED)
This repo implements a set of detection methods for adversarial example detection (AED). So far, it supports experiments on MNIST, CIFAR-10, and SVHN datasets with 7 detection methods: KDE, LID, NSS, FS, Magnet, NIC, and MultiLID. A brief description and reference of these methods can be found below. 

### Setting Paths
Open `setup_paths.py` and configure the paths and other settings for the detection methods.

### Train Model
To train a model, run `train_model.py -d=<dataset> -b=<batch_size> -e=<epochs>`.

### Generate Adversarial Example
To generate adversarial examples, run `generate_adv.py -d=<dataset>`. After running the program, adversarial examples will be automatically generated and saved for subsequent detection. After running the program, adversarial examples will be automatically generated and saved for subsequent detection. Additionally, the perturbation for $L_{\infty}$ are `epsilons = [8/256, 16/256, 32/256, 64/256, 80/256, 128/256]`, for L1 are `epsilons1 = [5, 10, 15, 20, 25, 30, 40]`, and for L2 are `epsilons2 = [0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]`.

### Detector
To run all the detectors, just execute `run_detectors.py`. If you want to run a specific detection method, execute `detect_{method_name}.py -d=<dataset>`, replacing {method_name} with the name of the method you wish to run. For example, `detect_multiLID.py -d=cifar`.

### Results
Here, we only report the Area Under Curve (AUC, %). Other performance results, like ACC, TP, TN, FP, and FN, can be acquired from the generated CSV file for each detector, by executing `collect_results.py`

（1）CIFAR10
| Attack                        | Parameters            | KDE   | LID   | NSS   | FS    | MagNet | NIC   | MultiLID |
|-------------------------------|-----------------------|-------|-------|-------|-------|--------|-------|----------|
| FGSM                          | $L_{\infty}$(ϵ=8/256)  | 75.82 | 62.57 | 92.7  | 63.27 | 94.49  | 100   | 97.58    |
| FGSM                          | $L_{\infty}$(ϵ=16/256) | 82.83 | 93.13 | 95.33 | 49.74 | 95.54  | 100   | 98.84    |
| FGSM                          | $L_{\infty}$(ϵ=32/256) | 93.18 | 95.5  | 95.21 | 36.97 | 95.55  | 100   | 99.04    |
| BIM                           | $L_{\infty}$(ϵ=8/256)  | 77.82 | 55.88 | 85.1  | 97.85 | 91.6   | 100   | 97.23    |
| BIM                           | $L_{\infty}$(ϵ=16/256) | 75.93 | 75.46 | 93.44 | 98.32 | 95.29  | 100   | 98.49    |
| BIM                           | $L_{\infty}$(ϵ=32/256) | 73.82 | 95.21 | 95.23 | 98.32 | 95.52  | 100   | 99.28    |
| PGD                           | $L_{\infty}$(ϵ=8/256)  | 79.14 | 56.89 | 86.62 | 98.18 | 92.85  | 100   | 97.18    |
| PGD                           | $L_{\infty}$(ϵ=16/256) | 76.01 | 77.84 | 94.21 | 98.32 | 95.45  | 100   | 98.54    |
| PGD                           | $L_{\infty}$(ϵ=32/256) | 75.57 | 95.4  | 95.33 | 98.32 | 95.52  | 100   | 99.34    |
| CW                            | $L_{\infty}$           | 94    | 49.57 | 58.44 | 45.04 | 61.62  | 100   | 96.96    |
| DeepFool                      | $L_2$                  | 96.68 | 50.11 | 52.75 | 43.82 | 52.09  | 100   | 96.77    |
| Spatial Transformation Attack | -                      | 75.71 | 91.65 | 94.45 | 24.08 | 90.2   | 100   | 99.96    |
| Square Attack                 | $L_{\infty}$           | 97.94 | 92.77 | 26.09 | 32.56 | 99.9   | 100   | 99.66    |
| Adversarial Patch             | -                      | 66.32 | 72.38 | 96.72 | 51.76 | 99.02  | 100   | 99.97    |

（2）SVHN
| Attack          | Parameters             | KDE   | LID   | NSS   | FS    | MagNet | NIC   | MultiLID |
|-----------------|------------------------|-------|-------|-------|-------|--------|-------|----------|
| FGSM            | $L_{\infty}$(ϵ=8/256)  | 68.4  | 77.77 | 99.75 | 81.54 | 87.1   | 100   | 99.69    |
| FGSM            | $L_{\infty}$(ϵ=16/256) | 69.14 | 89.91 | 99.99 | 87.26 | 97.16  | 100   | 99.78    |
| FGSM            | $L_{\infty}$(ϵ=32/256) | 74.27 | 98.45 | 99.19 | 93.48 | 99.67  | 100   | 99.79    |
| BIM             | $L_{\infty}$(ϵ=8/256)  | 47.7  | 69.98 | 99.3  | 98.42 | 82.08  | 100   | 99.67    |
| BIM             | $L_{\infty}$(ϵ=16/256) | 54.85 | 83.83 | 99.99 | 99.77 | 95.12  | 100   | 99.69    |
| BIM             | $L_{\infty}$(ϵ=32/256) | 58.61 | 96.39 | 100   | 99.99 | 99.11  | 100   | 99.72    |
| PGD             | $L_{\infty}$(ϵ=8/256)  | 48.14 | 70.76 | 99.54 | 99.13 | 84.64  | 100   | 99.66    |
| PGD             | $L_{\infty}$(ϵ=16/256) | 55.8  | 85.68 | 99.99 | 99.92 | 96.64  | 100   | 99.69    |
| PGD             | $L_{\infty}$(ϵ=32/256) | 59.31 | 97.11 | 100   | 100   | 99.52  | 100   | 99.69    |
| CW              | $L_{\infty}$           | 71.5  | 53.67 | 67.23 | 84.27 | 55.95  | 100   | 99.63    |
| DeepFool        | $L_2$                  | 71.42 | 51.96 | 49.8  | 78.04 | 51.6   | 100   | 99.63    |
| Spatial Transformation Attack             | -                      | 77.22 | 97.59 | 99.99 | 73.55 | 92.74  | 100   | 99.66    |
| Square Attack   | $L_{\infty}$           | 77.19 | 84.62 | 65.71 | 95.41 | 99.61  | 100   | 99.71    |
| Adversarial Patch | -                    | 41.77 | 74.92 | 99.99 | 74.63 | 92.97  | 100   | 99.69    |

（3）UNSWNB15
| Attack          | Parameters             | Logistic Regression   | 
|-----------------|------------------------|-------|
| Net Intrusion            | -  | 96.89  |



### Attack & Detection Methods
Attack methods: 

1. FGSM<sup>[8]</sup>: a one-step gradient sign attack method. **[one-step attack]** 

2. BIM<sup>[9]</sup>:  an iterative multi-step attack method with equally divided step size. **[multi-step attack]**

3. PGD<sup>[10]</sup>:   an interactive attack method with uniform initialization, large step size, and perturbation projection. **[the strongest first-order attack]**

4. CW<sup>[11]</sup>: an optimization-based attack framework that minimize the L2 perturbation magnitude, while targetting maximum classification error. **[L2 optimization attack]**

5. DeepFool<sup>[12]</sup>: a decision boundary based attack with adaptive perturabtion. **[boundary attack]**

6. Spatial Transformation Attack<sup>[13]</sup>: spatially transform the samples to be adversarial, different from other attacks that perturb the pixel values. **[spatial attack]**

7. Square Attack<sup>[14]</sup>: a score-based black-box L2 adversarial attack that selects localized square-shaped updates at random positions at each iteration. **[black-box attack]**

8. Adversarial Patch<sup>[15]</sup>: a patch with large adversarial perturbations attached to a random square/round area of the image. **[pyshical attack]**

Detection methods: 

1. [KDE](https://arxiv.org/pdf/1703.00410)<sup>[1]</sup>: KDE reveals that adversarial samples tend to deviate from the normal data manifold in the deep space, resulting in relatively lower kernel densities.

2. [LID](https://arxiv.org/pdf/1801.02613)<sup>[2]</sup>: This method extracts features from each intermediate layer of a deep neural network and employs the Local Intrinsic Dimensionality metric to detect adversarial samples.

3. [NSS](https://ieeexplore.ieee.org/document/9206959)<sup>[3]</sup>: This method proposes to characterize the AEs through the use of natural scene statistics.

4. [FS](https://arxiv.org/abs/1704.01155)<sup>[4]</sup>: This method employs feature squeezing to reduce the dimensionality of input samples and then detects adversarial samples based on the changes in the model's output before and after compression.

5. [MagNet](https://arxiv.org/abs/1705.09064)<sup>[5]</sup>: This method detects adversarial samples by assessing the ability to reconstruct normal samples while being unable to reconstruct adversarial samples. The AEs can be easily distinguished from those of normal samples using MSCN coefficients as the NSS tool. 

6. [NIC](https://www.cs.purdue.edu/homes/taog/docs/NDSS19.pdf)<sup>[6]</sup>: This method proposes a novel technique to extract DNN invariants and use them to perform runtime adversarial sample detection. 

7. [MultiLID](https://arxiv.org/pdf/2212.06776)<sup>[7]</sup>: Based on a re-interpretation of the LID measure and several simple adaptations, this method surpasses the state-of-the-art on adversarial detection.

## References
[1] Feinman R, Curtin R R, Shintre S, et al. Detecting adversarial samples from artifacts [J]. arXiv preprint arXiv:170300410, 2017.

[2] Ma X, Li B, Wang Y, et al. Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality[C]//International Conference on Learning Representations, 2018.

[3] Kherchouche A, Fezza S A, Hamidouche W, et al. Detection of adversarial examples in deep neural networks with natural scene statistics[C]//2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020: 1-7.

[4] Xu W, Evans D, Qi Y. Feature squeezing: Detecting adversarial examples in deep neural networks [J]. arXiv preprint arXiv:170401155, 2017.

[5] Meng D, Chen H. Magnet: a two-pronged defense against adversarial examples[C]// Proceedings of the 2017 ACM SIGSAC conference on computer and communications security, 2017.

[6] Ma S, Liu Y, Tao G, et al. Nic: Detecting adversarial samples with neural network invariant checking[C]// 26th Annual Network And Distributed System Security Symposium (NDSS 2019), 2019. Internet Soc.

[7] Lorenz P, Keuper M, Keuper J. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection[J]. arXiv preprint arXiv:2212.06776, 2022.

[8] Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[C]//International Conference on Learning Representations, 2015.

[9] Kurakin A, Goodfellow I J, Bengio S. Adversarial Examples in the Physical World[J]. Artificial Intelligence Safety and Security, 2018: 99-112.

[10] Madry A, Makelov A, Schmidt L, et al. Towards deep learning models resistant to adversarial attacks[C]//International Conference on Learning Representations, 2018.

[11] Carlini N, Wagner D. Towards evaluating the robustness of neural networks[C]//2017 IEEE symposium on security and privacy (sp). Ieee, 2017: 39-57.

[12] Moosavi-Dezfooli S M, Fawzi A, Frossard P. Deepfool: a simple and accurate method to fool deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2574-2582.

[13] Engstrom L, Tran B, Tsipras D, et al. Exploring the landscape of spatial robustness[C]//International Conference on Learning Representations, 2019.

[14] Andriushchenko M, Croce F, Flammarion N, et al. Square attack: a query-efficient black-box adversarial attack via random search[C]//European Conference on Computer Vision, 2020.

[15] Brown T B, Mané D, Roy A, et al. Adversarial patch[J]. arXiv preprint arXiv:1712.09665, 2017.
