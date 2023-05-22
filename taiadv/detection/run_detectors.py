import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
from common.util import *
from setup_paths import *

seed = 123

#run FS
for dataset in DATASETS:
    print("FS  --  dataset: {}  --  attack: all ".format(dataset))
    os.system('{}{}detect_fs.py -d={} -s={}'.format(env_param, detectors_dir, dataset, seed))

#run KDE
for dataset in DATASETS:
    ATTACKS = ATTACK[DATASETS.index(dataset)]
    for attack in ATTACKS:
        print("KDE  --  dataset: {}  --  attack: {} ".format(dataset, attack))
        os.system('{}{}detect_kde.py -d={} -a={} -s={}'.format(env_param, detectors_dir, dataset, attack, seed))

#run LID
for dataset in DATASETS:
    ATTACKS = ATTACK[DATASETS.index(dataset)]
    for attack in ATTACKS:
        print("LID  --  dataset: {}  --  attack: {} ".format(dataset, attack))
        os.system('{}{}detect_lid.py -d={} -a={} -k={} -s={}'.format(env_param, detectors_dir, dataset, attack, k_nn[DATASETS.index(dataset)], seed))

#run MagNet
for dataset in DATASETS:
    print("MagNet  --  dataset: {}  --  attack: all ".format(dataset))
    os.system('{}{}detect_magnet.py -d={} -s={}'.format(env_param, detectors_dir, dataset, seed))

#run NSS
for dataset in DATASETS:
    print("NSS  --  dataset: {}  --  attack: all ".format(dataset))
    os.system('{}{}detect_nss.py -d={} -s={}'.format(env_param, detectors_dir, dataset, seed))

# #run NIC
# for dataset in DATASETS:
#     print("NIC  --  dataset :: {}  --  attack :: all ".format(dataset))
#     os.system('{}{}detect_nic.py -d={} -s={}'.format(env_param, detectors_dir, dataset, seed))

# #run DNR
# for dataset in DATASETS[0:3]:
#     print("dataset :: {}  -- attack :: all ".format(dataset))
#     os.system('{}{}detect_dnr.py -d={}'.format(env_param, detectors_dir, dataset))

# #run SFAD
# for dataset in DATASETS:
#     print("dataset :: {}  -- attack :: all ".format(dataset))
#     os.system('{}{}detect_sfad.py -d={}'.format(env_param, detectors_dir, dataset))