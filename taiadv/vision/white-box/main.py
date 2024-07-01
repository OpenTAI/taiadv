import os
import util
import time
import argparse
import torch
import numpy as np
from datasets import DatasetGenerator
from attacks.attack_handler import Attacker
from tqdm.auto import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description="attack")
parser.add_argument('--dataset',type=str,default='CIFAR10')
parser.add_argument('--datapath', type=str, default='./')
parser.add_argument('--modelpath',type=str,default='./')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--attack_type',type=str,default='PGD')
parser.add_argument('--random_start',type=bool,default=True)
parser.add_argument('--noise',type=str,default="Uniform")
parser.add_argument('--num_restarts',type=int,default=4)
parser.add_argument('--step_size',type=float,default=2./255)
parser.add_argument('--num_steps',type=int,default=10)
parser.add_argument('--loss_f',type=str,default="mg")
parser.add_argument('--use_odi',type=bool,default=False)
parser.add_argument('--num_classes',type=int ,default=10)
parser.add_argument('--result_path',type=str,default='./re')

args = parser.parse_args()
args.eps = args.eps/255

def setup(rank,world_size):
    if world_size>1:
        dist.init_process_group('nccl',rank=rank,world_size=world_size)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def test(rank,world_size,data_loader,model):
    total = 0
    clean_total = 0
    adv_total = 0
    
    adversary = Attacker(model, attack_type = args.attack_type, eps=args.eps, random_start=args.random_start,
                noise = args.noise, num_restarts=args.num_restarts, step_size=args.step_size,
                num_steps=args.num_steps, loss_f=args.loss_f, use_odi=args.use_odi,
                num_classes=args.num_classes,verbose=True,bs=args.bs)
    
    start = time.time()
    
    for images,labels in tqdm(data_loader):
        #x_adv = images.clone()
        images,labels = images.to(rank),labels.to(rank)
        total += images.shape[0]
        with torch.no_grad():
            clean_logits = model(images)
            if isinstance(clean_logits, list): 
                clean_logits = clean_logits[-1]
        clean_pred = clean_logits.max(1)[1].detach()
        accs = clean_pred==labels
        if args.dataset ==  "CC1M":
            labels = clean_pred
            clean_total += images.shape[0]
        else:
            clean_total += accs.sum().item()
            
        for a in adversary.attacks_to_run:
            index = accs.nonzero()
            if index.numel() > 0:
                index = index.squeeze()
            else:
                continue
            x,y = images[index].clone(),labels[index].clone()
            if len(x.shape)!=4:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
            x_adv,adv_acc = a.perturb(x,y)
            accs[index] = adv_acc
            #x_adv[index] = x_adv.cpu()
        adv_total += accs.sum().item()
        if rank == 0:
            print(("Clean:%d/%d Clean Acc: %.2f Adv: %d/%d Adv_Acc: %.2f")%(clean_total,total,clean_total/total*100,adv_total,total,adv_total/total*100))
        else:
            print("???")
    if world_size>1:
        total_tensor = torch.tensor(total).to(rank)
        clean_total_tensor = torch.tensor(clean_total).to(rank)
        adv_total_tensor = torch.tensor(adv_total).to(rank)
        dist.all_reduce(total_tensor,op=dist.ReduceOp.SUM)
        dist.all_reduce(clean_total_tensor,op=dist.ReduceOp.SUM)
        dist.all_reduce(adv_total_tensor,op=dist.ReduceOp.SUM)
        total = total_tensor.item()
        clean_total = clean_total_tensor.item()
        adv_total = adv_total_tensor.item()
    end = time.time()
    if rank == 0:
        clean_accuracy, robust_accuracy = round(clean_total/total*100,2), round(adv_total/total*100,2)

        print(f"clean_accuracy:{clean_accuracy}")
        print(f"robust_accuracy:{robust_accuracy}")
        cost = end - start
        print(f"cost:{cost}")
        payload = {
            'attack_type':args.attack_type,
            'num_steps':args.num_steps,
            'loss_f':args.loss_f,
            'clean_acc': clean_accuracy,
            'adv_acc': robust_accuracy,
            'cost': cost,
        }
        args.result_path = os.path.join(args.result_path,args.dataset)
        util.build_dirs(args.result_path)
        filename = '%s.json' % (args.attack_type)
        filename = os.path.join(args.result_path, filename)
        print(filename)
        util.save_json(payload, filename)
        return
    

def main(rank,world_size):
    setup(rank,world_size)
    print(rank)
    model = torch.load(args.modelpath)
    model = model.to(rank)
    model.eval()
    
    #if world_size>1:
    #    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],output_device=rank)
        
    for param in model.parameters():
        param.requires_grad = False
        
    data = DatasetGenerator(eval_bs=args.bs,test_d_type=args.dataset,test_path=args.datapath)
    
    if world_size>1:
        data_loader = data.get_loader(is_sampler=True,rank=rank,world_size=world_size)
    else:
        data_loader = data.get_loader(is_sampler=False)
    
    test(rank,world_size,data_loader,model)
    
    cleanup()
    

if __name__ == '__main__':
    rank = int(os.getenv('RANK',0))
    world_size = int(os.getenv('WORLD_SIZE',1))
    main(rank,world_size)
