# Adversarial-Example-Detection
This is a project for detecting adversarial examples. It supports the MNIST, CIFAR-10, and SVHN datasets, and currently includes detection methods such as KDE, LID, NSS, FS, Magnet, NIC, and MultiLID.

## Setting Paths
Open `setup_paths.py` and configure the paths and other settings for the detection methods.

## Train model
To train a model, run `train_model.py -d=<dataset> -b=<batch_size> -e=<epochs>`.

## Generate adversarial example
To generate adversarial examples, run `generate_adv.py -d=<dataset>`.

## Detection
To run all the detector, just execute `run_detectors.py`. If you want to run a specific detection method, execute `detect_{method_name}.py`, replacing {method_name} with the name of the method you wish to run.

## Results
| Attack                        | KDE_DR | LID_DR | NSS_DR | FS_DR | MagNet_DR | NIC_DR | MultiLID_DR |
|-------------------------------|--------|--------|--------|-------|-----------|--------|-------------|
| fgsm_0.03125                  | 66.47  | 50.0   | 84.33  | 52.51 | 51.62     | 94.32  | 92.81       |
| fgsm_0.0625                   | 63.96  | 78.98  | 92.87  | 49.84 | 94.26     | 94.79  | 93.46       |
| fgsm_0.125                    | 61.44  | 83.97  | 92.85  | 49.27 | 94.35     | 94.82  | 93.86       |
| bim_0.03125                   | 69.43  | 50.11  | 67.42  | 93.18 | 50.45     | 90.55  | 92.9        |
| bim_0.0625                    | 69.05  | 66.21  | 86.82  | 93.98 | 65.38     | 92.37  | 93.54       |
| bim_0.125                     | 69.01  | 92.1   | 92.6   | 93.99 | 94.15     | 94.44  | 94.05       |
| pgdi_0.03125                  | 71.04  | 50.11  | 69.85  | 93.81 | 50.62     | 90.72  | 92.86       |
| pgdi_0.0625                   | 70.95  | 68.06  | 89.41  | 93.99 | 92.56     | 94.07  | 93.59       |
| pgdi_0.125                    | 70.37  | 92.83  | 92.78  | 93.99 | 94.15     | 94.68  | 94.46       |
| cwi                           | 75.34  | 50.0   | 51.73  | 48.16 | 50.08     | 87.74  | 98.02       |
| deepfool                      | 81.68  | 50.0   | 50.44  | 48.35 | 50.02     | 93.11  | 98.06       |
| spatial transofrmation attack | 68.88  | 83.77  | 78.01  | 47.71 | 50.67     | 91.33  | 99.67       |
| square attack                 | 75.36  | 80.76  | 48.89  | 47.72 | 98.58     | 94.67  | 99.22       |
| adversarial patch             | 52.43  | 64.11  | 87.39  | 48.67 | 57.97     | 94.58  | 99.76       |
