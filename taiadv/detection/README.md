# Adversarial-Example-Detection

 adversarial example detection
 dataset: minist, cifar10
 detection: FS, KDE, LID, Magnet, NSS, NIC

## Set paths
Open setup_paths.py and set the paths and other detection method settings.

## Train model
Run train_model.py -d=<dataset>

## Generate adversarial example
Run generate_adv.py -d=<dataset>

## Detection
To run all the detector, just execute run_detectors.py.