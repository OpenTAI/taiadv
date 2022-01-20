# Examples of using OpenTAIAdv
The goal of our **OpenTAIAdv** is to provide a simple framework for researching adversarial attacks and defence.

---
## Quickstart examples
For a quick start, we have provided the following examples:
- train.py
- *others coming soon*

### Adversarial Training with OpenTAIAdv
Here are some descriptions of the provided **train.py**
The experiments configuration files are stored under [configs folders](configs/). We provided configurations for [Standard Adversarial Training](https://arxiv.org/abs/1706.06083), [TRADES](https://arxiv.org/pdf/1901.08573.pdf), [MART](https://openreview.net/forum?id=rklOg6EFwS), using WRN-34-10 model on CIFAR-10 with L_inf.
To run the example to train a robust model, please follow the following:
```python
    python train.py --exp_path /PATH/TO/YOUR/EXPERIMENT/FOLDER \
                    --exp_name wrn34x10_sat                    \
                    --exp_configs configs/                     \
```
 - *exp_name* option can be replaced with other experiments configuration files stored under the configs folder.
 - *exp_path* is where you want to store the experiment's files, such as checkpoints and logs
 - *exp_config* is the path to the folder where the configuration files are stored
 - *data_parallel* if you want to use data_parallel **torch.nn.DataParallel**.
 - Adv Attack Options (*eps*, *num_steps*, *step_size*, *attack_choice*) can be replace, this script will run a adversarial attack evaluation on validation set in every epochs to validate the performance.
