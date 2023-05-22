# Common
checkpoints_dir = './checkpoints/'
adv_data_dir = './adv_data/'
adv_data_gray_dir = './adv_data/gray/'
DATASETS = ['mnist', 'cifar'] # ['mnist', 'cifar', 'imagenet']
ATTACK = [['fgsm_0.125', 'fgsm_0.25', 'fgsm_0.3125', \
            'bim_0.125', 'bim_0.25', 'bim_0.3125', \
            'pgd1_10', 'pgd1_15', 'pgd1_20', \
            'pgd2_1', 'pgd2_1.5', 'pgd2_2', \
            'pgdi_0.125', 'pgdi_0.25',\
            'cwi', \
            'df',\
            'sta',\
            'sa'
            ],
            ['fgsm_0.03125', 'fgsm_0.0625', 'fgsm_0.125',\
            'bim_0.03125', 'bim_0.0625', \
            'pgd1_5', 'pgd1_10', 'pgd1_15', \
            'pgd2_0.25', 'pgd2_0.3125', 'pgd2_0.5',\
            'pgdi_0.03125', 'pgdi_0.0625', \
            'cwi', \
            'df',\
            'sta',\
            'sa'
            ],
            ['fgsm_0.03125', 'fgsm_0.0625',\
            'bim_0.03125', 'bim_0.0625', 'bim_0.125',\
            'pgd1_15', 'pgd1_20', 'pgd1_25', \
            'pgd2_0.5', 'pgd2_1',\
            'pgdi_0.03125', 'pgdi_0.0625', \
            'cwi', \
            'df',\
            'sta',\
            'sa',\
            # 'zoo'
            ]
        ]

ALL_ATTACKS = ['fgsm_0.03125', 'fgsm_0.0625', 'fgsm_0.125', 'fgsm_0.25', 'fgsm_0.3125',\
            'bim_0.03125', 'bim_0.0625', 'bim_0.125', 'bim_0.25', 'bim_0.3125',\
            'pgd1_5', 'pgd1_10', 'pgd1_15', 'pgd1_20', 'pgd1_25',\
            'pgd2_0.25','pgd2_0.3125', 'pgd2_0.5', 'pgd2_1', 'pgd2_1.5', 'pgd2_2',\
            'pgdi_0.03125', 'pgdi_0.0625', 'pgdi_0.125', 'pgdi_0.25',\
            'cwi', \
            'hca_0.03125', 'hca_0.0625', 'hca_0.125', 'hca_0.3125', 'hca_0.5',\
            'df',\
            'sa', 'hop', 'sta',\
            'zoo'
           ]
fieldnames = ['type',	'nsamples',	'acc_suc',	'acc',	'tpr',	'fpr',	'tp',	'ap',	'fb',	'an',	'tprs',	'fprs',	'auc']
env_param = 'env /remote-home/wangxin/venv/detect/bin/python -- ' 
detectors_dir = './'
results_path = './results/'
#-------------------------- detect KDE
# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00, 'imagenet': 0.26}
#[0.1, 0.16681005372000587, 0.2782559402207124, 0.46415888336127786, 0.774263682681127, 1.291549665014884, 2.1544346900318834, 3.593813663804626, 5.994842503189409, 10.0]
kde_results_dir = './results/kde/'
kde_results_gray_dir = './results/kde/gray/'

#-------------------------- detect LID
k_nn = [20, 30, 20, 30]
lid_results_dir = './results/lid/'
lid_results_gray_dir = './results/lid/gray/'

#-------------------------- detect MagNet
magnet_results_dir = './results/magnet/'
magnet_results_gray_dir = './results/magnet/gray/'

#-------------------------- detect FS
fs_results_dir = './results/fs/'
fs_results_gray_dir = './results/fs/gray/'

#-------------------------- detect NSS
pgd_percent = [[0.02, 0.1, 0.18, 0.3, 0.3, 0.1], [0.1, 0.3, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]]
nss_results_dir = './results/nss/'
nss_results_gray_dir = './results/nss/gray/'

#-------------------------- detect NIC
nic_results_dir = './results/nic/'
nic_results_gray_dir = './results/nic/gray/'
nic_layers_dir = './results/nic/layers/'
nic_layers_gray_dir = './resultsx/nic/layers/gray/'


