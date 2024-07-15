import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from easydict import EasyDict as edict 
from datetime import datetime

from models import create_model_by_rank, model_list, no_head_model_list
from data import get_data_loader, num_classes_dict
import misc
import loss
# https://github.com/mlomnitz/DiffJPEG
from DiffJPEG import DiffJPEG
from utils import transform_and_stack

# set random seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


diffJPEG = DiffJPEG(224,224,differentiable=True, quality=75)
class NoiseModule(nn.Module):  
    def __init__(self, shape, cfg):  
        super(NoiseModule, self).__init__()  
        self.noise = nn.Parameter(torch.zeros(shape)) 
        if cfg.random:
            self.noise.data.uniform_(-cfg.epsilon, cfg.epsilon)
        self.jpeg_resistant = cfg.jpeg_resistant
  
    def forward(self, model, x) -> torch.Tensor : 
        x_adv = x + self.noise
        x_adv = torch.clamp(x_adv, 0, 1)
        if self.jpeg_resistant:
            x_adv = diffJPEG(x_adv) 
        return model(x_adv)
      
class NoiseModuleV2(nn.Module):  
    def __init__(self, shape, cfg):  
        super().__init__()  
        self.noise = nn.Parameter(torch.zeros(shape)) 
        if cfg.random:
            self.noise.data.uniform_(-1, 1)
        self.jpeg_resistant = cfg.jpeg_resistant
        self.eps = cfg.epsilon
        self.forward_feat = cfg.loss == 'cos'
  
    def forward(self, model, x) -> torch.Tensor : 
        x_adv = x + torch.tanh(self.noise) * self.eps
        x_adv = torch.clamp(x_adv, 0, 1)
        if self.jpeg_resistant:
            x_adv = diffJPEG(x_adv) 
        if self.forward_feat:
            return model.forward_features(x_adv)
        else:
            return model(x_adv)
          
# not stacked x
class NoiseModuleV3(nn.Module):  
    def __init__(self, x_list: list, cfg):  
        super().__init__()  
        self.noise = nn.ParameterList([nn.Parameter(torch.zeros_like(x, dtype=torch.float32)) for x in x_list])
        if cfg.random:
            for noise_i in self.noise:
                noise_i.data.uniform_(-1, 1)
        self.jpeg_resistant = cfg.jpeg_resistant
        self.eps = cfg.epsilon
        self.forward_feat = cfg.loss == 'cos'
  
    def forward(self, model, x_list: list) -> torch.Tensor : 
        # print(x_list[0].device, self.noise[0].device)
        x_adv_list = [x + torch.tanh(noise) * self.eps for x, noise in zip(x_list, self.noise)]
        x_adv = transform_and_stack(x_adv_list, aug = True)
        x_adv = torch.clamp(x_adv, 0, 1)
        if self.jpeg_resistant:
            x_adv = diffJPEG(x_adv) 
        if self.forward_feat:
            return model.forward_features(x_adv)
        else:
            return model(x_adv)  


def get_config():
    parser = argparse.ArgumentParser(description="Configuration for attack")
    parser.add_argument('--dataset', type=str, default='cc1m', help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='../data', help='Path to the data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--random', action='store_true', default=True, help='Use random initialization')
    parser.add_argument('--loss', type=str, default='cos', help='Loss function')
    parser.add_argument('--epsilon', type=float, default=10/255, help='Epsilon for attack')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of steps for the attack')
    parser.add_argument('--step_size', type=float, default=1, help='Step size for the attack')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer')
    parser.add_argument('--jpeg_resistant', action='store_true', default=True, help='JPEG resistant')

    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--dist_on_itp', default=None, help='Distributed on ITP')
    parser.add_argument('--rank', default=None, help='Rank of the process')
    parser.add_argument('--world_size', default=None, help='World size')
    parser.add_argument('--gpu', default=None, help='GPU id')
    parser.add_argument('--dist_url', default=None, help='URL used to set up distributed training')
    
    cfg = parser.parse_args()
    return cfg


label_cnt = {}
def get_label_cnt(label):
    if label not in label_cnt:
        label_cnt[label] = -1
    label_cnt[label] += 1
    return label_cnt[label]

def get_lex(num_class):
    strList = []
    for i in range(num_class):
        strList.append(str(i))
    return sorted(strList)

# for eval
total = 0
correct = 0
def eval(adv_image, y, model, f):
    global total, correct
    # adv_iamge is list or tensor
    if type(adv_image) is list:
        adv_image = transform_and_stack(adv_image)
    if type(y) is list:
        y = torch.tensor(y, device=adv_image.device)
        y = y.long()
    outputs = model(adv_image)
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()
    f.write(f'acc {correct / total*100:.2f} {datetime.now()}\n' )
    print(f'acc {correct / total*100:.2f}', datetime.now(), flush=True)

def main(cfg):
    global diffJPEG

    lex = get_lex(num_classes_dict[cfg.dataset] if cfg.dataset in num_classes_dict else 1)
    
    misc.init_distributed_mode(cfg)
    if misc.is_main_process():
        from pprint import pprint
        pprint(cfg)
        print(datetime.now())
    r = misc.get_rank()
    f = open(f'log_{r}.txt', 'w')
    f.write(f'rank {r}\n')
    print(r)
    device = torch.device(f'cuda:{r}')

    # model = create_model_by_rank(r, num_classes_dict[cfg.dataset] if cfg.dataset in num_classes_dict else 0)
    model = create_model_by_rank(r, 1000)
    model = model.eval()
    model = model.to(device)
    diffJPEG = diffJPEG.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    data_loader = get_data_loader(cfg.dataset, root_folder=cfg.data_path, batch_size= cfg.batch_size, aug = transforms.ToTensor(), separate=True)


    for i,(x, y) in enumerate(data_loader):
        # if i == 10:
        #     break
        # x = x.to(device)
        for j in range(len(x)):
            x[j] = x[j].to(device)
        if type(y) == torch.Tensor:
            y = y.to(device)
        # f.write(f'batch {i} {x[0][0][:10]=}\n')
        noiseModule = NoiseModuleV3(x, cfg)
        noiseModule = noiseModule.to(device)
        noiseModule = torch.nn.parallel.DistributedDataParallel(noiseModule)
        # set optimizer
        if cfg.optim == 'sgd':
            optimizer = optim.SGD(noiseModule.parameters(), lr=cfg.step_size, maximize=True, momentum=1)
        elif cfg.optim == 'adam':
            optimizer = optim.Adam(noiseModule.parameters(), lr=cfg.step_size, maximize=True)
        else:
            raise NotImplementedError()
        # rank specific
        loss_fn = loss.__dict__[cfg.loss]


        for iter in range(cfg.num_steps):
            loss_val = loss_fn(x, y, model, noiseModule)
            optimizer.zero_grad()
            loss_val.backward()
            # noiseModule.module.noise.grad.data = torch.sign(noiseModule.module.noise.grad.data)
            optimizer.step()
            # boundary projection
            # noiseModule.module.noise.data = torch.clamp(noiseModule.module.noise.data, -cfg.epsilon, cfg.epsilon)
            # assert noiseModule.module.noise.max() < 11/255

        if cfg.num_steps > 0:
            f.write(f'{loss_val.item()=}\n')
            # f.write(f'{next(noiseModule.parameters()).grad[0][0][:10]=}\n')
            f.flush()

        # adv_image = torch.clamp((noiseModule.module.noise + x), 0, 1).detach()
        # adv_image = torch.clamp((torch.tanh(noiseModule.module.noise)*cfg.epsilon + x), 0, 1).detach()
        adv_image = [torch.clamp((torch.tanh(noise)*cfg.epsilon + x_i), 0, 1).detach() for x_i, noise in zip(x, noiseModule.module.noise)]
        if cfg.dataset in ['imagenet', 'cifar10'] and model_list[r] not in no_head_model_list:
            eval(adv_image, y, model, f)
        else:
            print(f'processing batch {i}', datetime.now())

        # save adv image
        if misc.is_main_process():
            for idx, image in enumerate(adv_image):
                label = y[idx]
                if type(label) == torch.Tensor:
                    label = label.item()

                if cfg.dataset == 'imagenet':
                    dir2 = f"imagenet_2012/val/{lex[label]}"
                elif cfg.dataset == 'cifar10':
                    dir2 = f"cifar10/{lex[label]}"
                else:
                    # 'coco2017', 'market1501', 'ade20k', 'cc1m', 'chexpert'
                    dir2 = cfg.dataset

                dir = f'adv_image/tmp_model{cfg.world_size}_{cfg.loss}/{dir2}'
                os.makedirs(dir, exist_ok=True)
                if cfg.dataset in ['imagenet', 'cifar10']:  
                    image_path = f'{dir}/image_{get_label_cnt(label)}.JPEG'
                else:
                    image_path = f'{dir}/{label}'
                    image_dir = os.path.dirname(image_path)
                    if image_dir and not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                torchvision.utils.save_image(image, image_path)
            print(label)


    f.close()
    print('ok', datetime.now())

def main_submitit():
    import submitit
    executor = submitit.AutoExecutor(folder="logs_submitit")
    num_tasks = 8
    executor.update_parameters(
        timeout_min=60*48, 
        mem_gb=32*num_tasks,
        gpus_per_node=num_tasks,
        tasks_per_node=num_tasks,
        cpus_per_task=2,
        nodes=1,
        slurm_partition="fvl",
        slurm_qos = 'high',
        # slurm_partition="scavenger",
        # slurm_qos = 'scavenger',
        # slurm_srun_args = ['--nodelist','fvl16'],
        )
    cfg = get_config()
    # cfg.world_size = num_tasks
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(10000, 20000))
    os.environ["WORLD_SIZE"] = str(num_tasks)
    job = executor.submit(main, cfg)
    print(job.job_id)

if __name__ == '__main__':
    # main_submitit()
    cfg = get_config()
    main(cfg)