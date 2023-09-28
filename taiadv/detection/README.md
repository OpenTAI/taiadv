# Adversarial-Example-Detection
This is a project for detecting adversarial examples. It supports the MNIST, CIFAR-10, and SVHN datasets, and currently includes detection methods such as KDE[1], LID[2], NSS[3], FS[4], Magnet[5], NIC[6], and MultiLID[7].

Attack method: fgsm, bim, pgd, cw, deepfool, spatial transofrmation attack, square attack, adversarial patch

Detect method: 

(1)[KDE](https://arxiv.org/pdf/1703.00410): KDE reveals that adversarial samples tend to deviate from the normal data manifold in the deep space, resulting in relatively lower kernel densities.

(2)[LID](https://arxiv.org/pdf/1801.02613): This method extracts features from each intermediate layer of a deep neural network and employs the Local Intrinsic Dimensionality metric to detect adversarial samples.

(3)[NSS](https://ieeexplore.ieee.org/document/9206959): This method propose to characterize the AEs through the use of natural scene statistics.

(4)[FS](https://arxiv.org/abs/1704.01155): This method employs feature squeezing to reduce the dimensionality of input samples and then detects adversarial samples based on the changes in the model's output before and after compression.

(5)[MagNet](https://arxiv.org/abs/1705.09064): This method detects adversarial samples by assessing the ability to reconstruct normal samples while being unable to reconstruct adversarial samples. The AEs can be easily distinguished from those of normal samples using MSCN coefficients as NSS tool. 

(6)[NIC](https://www.cs.purdue.edu/homes/taog/docs/NDSS19.pdf): This method propose a novel technique to extract DNN invariants and use them to perform runtime adversarial sample detection. 

(7)[MultiLID](https://arxiv.org/pdf/2212.06776): Based on a re-interpretation of the LID measure and several simple adaptations, this method surpass the state-of-the-art on adversarial detection.

## Setting Paths
Open `setup_paths.py` and configure the paths and other settings for the detection methods.

## Train model
To train a model, run `train_model.py -d=<dataset> -b=<batch_size> -e=<epochs>`.

## Generate adversarial example
To generate adversarial examples, run `generate_adv.py -d=<dataset>`. After running the program, adversarial examples will be automatically generated and saved for subsequent detection.After running the program, adversarial examples will be automatically generated and saved for subsequent detection. Additionally, the perturbation for Linf are `epsilons = [8/256, 16/256, 32/256, 64/256, 80/256, 128/256]`, for L1 are `epsilons1 = [5, 10, 15, 20, 25, 30, 40]`, and for L2 are `epsilons2 = [0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]`.

## Detection
To run all the detector, just execute `run_detectors.py`. If you want to run a specific detection method, execute `detect_{method_name}.py -d=<dataset>`, replacing {method_name} with the name of the method you wish to run. For example, `detect_multiLID.py -d=cifar`.

## Results
In this website, we only report the detection rate (DR). Other performance results, like TP, TN, FP, and FN, can be accquired from the genenerated CSV file for each detector, by execute `collect_results.py`
| Attack                        | KDE    | LID    | NSS    | FS    | MagNet    | NIC    | MultiLID    |
|-------------------------------|--------|--------|--------|-------|-----------|--------|-------------|
| fgsm_0.03125                  | 66.47  | 50.0   | 84.33  | 52.51 | 69.58     | 94.32  | 92.81       |
| fgsm_0.0625                   | 63.96  | 78.98  | 92.87  | 49.84 | 94.31     | 94.79  | 93.46       |
| fgsm_0.125                    | 61.44  | 83.97  | 92.85  | 49.27 | 94.33     | 94.82  | 93.86       |
| bim_0.03125                   | 69.43  | 50.11  | 67.42  | 93.18 | 52.25     | 90.55  | 92.9        |
| bim_0.0625                    | 69.05  | 66.21  | 86.82  | 93.98 | 93.93     | 92.37  | 93.54       |
| bim_0.125                     | 69.01  | 92.1   | 92.6   | 93.99 | 94.11     | 94.44  | 94.05       |
| pgdi_0.03125                  | 71.04  | 50.11  | 69.85  | 93.81 | 53.52     | 90.72  | 92.86       |
| pgdi_0.0625                   | 70.95  | 68.06  | 89.41  | 93.99 | 94.08     | 94.07  | 93.59       |
| pgdi_0.125                    | 70.37  | 92.83  | 92.78  | 93.99 | 94.11     | 94.68  | 94.46       |
| cwi                           | 75.34  | 50.0   | 51.73  | 48.16 | 50.28     | 87.74  | 98.02       |
| deepfool                      | 81.68  | 50.0   | 50.44  | 48.35 | 50.05     | 93.11  | 98.06       |
| spatial transofrmation attack | 68.88  | 83.77  | 78.01  | 47.71 | 52.41     | 91.33  | 99.67       |
| square attack                 | 75.36  | 80.76  | 48.89  | 47.72 | 98.52     | 94.67  | 99.22       |
| adversarial patch             | 52.43  | 64.11  | 87.39  | 48.67 | 80.15     | 94.58  | 99.76       |

## Citation

## References
[1] Feinman R, Curtin R R, Shintre S, et al. Detecting adversarial samples from artifacts [J]. arXiv preprint arXiv:170300410, 2017.

[2] Ma X, Li B, Wang Y, et al. Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality[C]//International Conference on Learning Representations, 2018.

[3] Kherchouche A, Fezza S A, Hamidouche W, et al. Detection of adversarial examples in deep neural networks with natural scene statistics[C]//2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020: 1-7.

[4] Xu W, Evans D, Qi Y. Feature squeezing: Detecting adversarial examples in deep neural networks [J]. arXiv preprint arXiv:170401155, 2017.

[5] Meng D, Chen H. Magnet: a two-pronged defense against adversarial examples[C]// Proceedings of the 2017 ACM SIGSAC conference on computer and communications security, 2017.

[6] Ma S, Liu Y, Tao G, et al. Nic: Detecting adversarial samples with neural network invariant checking[C]// 26th Annual Network And Distributed System Security Symposium (NDSS 2019), 2019. Internet Soc.

[7] Lorenz P, Keuper M, Keuper J. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection[J]. arXiv preprint arXiv:2212.06776, 2022.


