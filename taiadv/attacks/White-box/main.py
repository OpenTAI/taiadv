import os
import util
import time
import argparse
import torch
import numpy as np
from datasets import DatasetGenerator
from attacks.attack_handler import Attacker
from tqdm.auto import tqdm

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description="attack")
parser.add_argument('--dataset',type=str,default='CIFAR10')
parser.add_argument('--datapath', type=str, default='/remote-home/xieyong/attack/CV/VIT/Attack/datasets/')
parser.add_argument('--modelpath',type=str,default='/remote-home/xieyong/attack/CV/VIT/Attack/models/checkpoints/CIFAR10/top10/Gowal2020Uncovering_70_16_extra.pth')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--attack_type',type=str,default='PGD')
parser.add_argument('--random_start',type=bool,default=True)
parser.add_argument('--noise',type=str,default="Uniform")
parser.add_argument('--num_restarts',type=int,default=1)
parser.add_argument('--step_size',type=float,default=2./255)
parser.add_argument('--num_steps',type=int,default=0)
parser.add_argument('--loss_f',type=str,default="mg")
parser.add_argument('--use_odi',type=bool,default=False)
parser.add_argument('--num_classes',type=int ,default=10)
parser.add_argument('--result_path',type=str,default='./re')

args = parser.parse_args()
args.eps = args.eps/255

def main():
    data = DatasetGenerator(eval_bs=args.bs,test_d_type=args.dataset,test_path=args.datapath)
    data_loader = data.get_loader()
    
    model = torch.load(args.modelpath)
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    start = time.time()
 
    adversary = Attacker(model, attack_type = args.attack_type, eps=args.eps, random_start=args.random_start,
                noise = args.noise, num_restarts=args.num_restarts, step_size=args.step_size,
                num_steps=args.num_steps, loss_f=args.loss_f, use_odi=args.use_odi,
                num_classes=args.num_classes,verbose=True,bs=args.bs)
    
    total = 0
    clean_total = 0
    adv_total = 0
    for images,labels in tqdm(data_loader):
        #x_adv = images.clone()
        images,labels = images.to(device),labels.to(device)
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
            index = accs.nonzero().squeeze()
            x,y = images[index].clone(),labels[index].clone()
            x_adv,adv_acc = a.perturb(x,y)
            accs[index] = adv_acc
            #x_adv[index] = x_adv.cpu()
        adv_total += accs.sum().item()
        print(("Clean:%d/%d Clean Acc: %.2f Adv: %d/%d Adv_Acc: %.2f")%(clean_total,total,clean_total/total*100,adv_total,total,adv_total/total*100))
            
    clean_accuracy, robust_accuracy = round(clean_total/total*100,2), round(adv_total/total*100,2)

    print(f"clean_accuracy:{clean_accuracy}")
    print(f"robust_accuracy:{robust_accuracy}")
    end = time.time()
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

if __name__ == '__main__':
    main()
