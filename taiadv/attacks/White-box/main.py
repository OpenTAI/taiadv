import os
import util
import time
import argparse
import torch
import numpy as np
from datasets import 
from attacks.Attack import Attacker

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description="attack")
parser.add_argument('--dataset',type=str,default='ImageNet')
parser.add_argument('--datapath', type=str, default='/mnt/lustre/share/images')
parser.add_argument('--modelpath',type=str,default='./models/IMAGENET/Singh2023Revisiting_ConvNeXt-S-ConvStem.pth')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=100)
parser.add_argument('--attack_type',type=str,default='PMA')
parser.add_argument('--random_start',type=bool,default=True)
parser.add_argument('--noise',type=str,default="Uniform")
parser.add_argument('--num_restarts',type=int,default=1)
parser.add_argument('--step_size',type=float,default=2./255)
parser.add_argument('--num_steps',type=int,default=100)
parser.add_argument('--loss_f',type=str,default="pm")
parser.add_argument('--use_odi',type=bool,default=False)
parser.add_argument('--num_classes',type=int ,default=1000)
parser.add_argument('--result_path',type=str,default='./results')

args = parser.parse_args()
args.eps = args.eps/255

def main():
    util.build_dirs(args.result_path)
    model = torch.load(args.modelpath)
    model = model.to(device)
    model.eval()

    data = Dataset_Processor(self.dataset,self.datapath)
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    start = time.time()
    adversary = Attacker(model, attack_type = args.attack_type, eps=args.eps, random_start=args.random_start,
             noise = args.noise, num_restarts=args.num_restarts, step_size=args.step_size,
             num_steps=args.num_steps, loss_f=args.loss_f, use_odi=args.use_odi,
             num_classes=args.num_classes,verbose=True, x_test=x_test, y_test=y_test, bs=args.bs )
    rs = adversary.evaluate()
    clean_accuracy, robust_accuracy, x_adv = rs
    end = time.time()
    cost = end - start

    print(f"clean_accuracy:{clean_accuracy*100:.2f}")
    print(f"robust_accuracy:{robust_accuracy*100:.2f}")
    print(f"cost:{cost}")
    
    payload = {
        'eps':args.eps,
        'bs': args.bs,
        'random_start':args.random_start,
        'noise':args.noise,
        'num_restarts':args.num_restarts,
        'attack_type':args.attack_type,
        'num_steps':args.num_steps,
        'step_size':args.step_size,
        'loss_f':args.loss_f,
        'use_odi':args.use_odi,
        'clean_acc': clean_accuracy,
        'adv_acc': robust_accuracy,
        'cost': cost,
    }
    filename = '%s_%s.json' % (args.attack_type,args.mark)
    filename = os.path.join(args.result_path+'/'+args.dataset+'/100', filename)
    util.save_json(payload, filename)
    return

if __name__ == '__main__':
    main()

