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

|   Attack                        |   KDE_DR  |   LID_DR  |   NSS_DR  |   FS_DR  |   MagNet_DR  |   NIC_DR  |   MultiLID_DR  |
|---------------------------------|-----------|-----------|-----------|----------|--------------|-----------|----------------|
|   fgsm_0.03125                  |   57.93   |   50.0    |   84.33   |   52.51  |   51.62      |   94.32   |   92.81        |
|   fgsm_0.0625                   |   56.43   |   78.98   |   92.87   |   49.84  |   94.26      |   94.79   |   93.46        |
|   bim_0.03125                   |   52.74   |   50.11   |   67.42   |   93.18  |   50.45      |   94.71   |   92.9         |
|   bim_0.0625                    |   50.0    |   66.21   |   86.82   |   93.98  |   65.38      |   91.71   |   93.54        |
|   pgdi_0.03125                  |   52.34   |   50.11   |   69.85   |   93.81  |   50.62      |   90.98   |   92.86        |
|   pgdi_0.0625                   |   50.0    |   68.06   |   89.41   |   93.99  |   92.56      |   93.39   |   93.59        |
|   pgdi_0.125                    |   72.84   |   92.83   |   92.78   |   93.99  |   94.15      |   94.29   |   94.46        |
|   cwi                           |   71.59   |   50.0    |   51.73   |   48.16  |   50.08      |   93.47   |   98.02        |
|   deep fool                     |   71.88   |   50.0    |   50.44   |   48.35  |   50.02      |   93.7    |   98.06        |
|   spatial transofrmation attack |   68.85   |   83.77   |   78.01   |   47.71  |   50.67      |   91.58   |   99.67        |
|   square attack                 |   49.98   |   80.76   |   48.89   |   47.72  |   98.58      |   94.72   |   99.22        |
|   adversarial patch             |   55.46   |   64.11   |   87.39   |   48.67  |   57.97      |   90.53   |   99.76        |
