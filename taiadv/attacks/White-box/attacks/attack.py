import torch
import numpy as np
from . import PGD
from . import autopgd_pt
from . import MD
from . import fab_pt
from autoattack import square,fab_pt
from .utils import adv_check_and_update
from tqdm.auto import tqdm
import pdb
import time
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Attacker():
    def __init__(self, model, attack_type = 'PGD_attack', eps=8./255, random_start=False,
                 noise = 'Uniform', num_restarts=1, step_size=2./255, bs = 32,
                 num_steps=100, loss_f='CE', use_odi=False,num_classes=10,
                 verbose=True, x_test=None, y_test=None, logger=None):
        self.model = model
        self.attack_type = attack_type
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.use_odi = use_odi
        self.noise = noise
        self.loss_f = loss_f
        self.num_classes = num_classes
        self.x_test = x_test
        self.y_test = y_test
        self.bs = bs
        self.logger = logger
        self.verbose = verbose
        
        self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type=self.loss_f)
        
        self.mtpgd = PGD.MTPGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type=self.loss_f)
        
        self.apgd = autopgd_pt.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss=self.loss_f, eot_iter=1, rho=.75, verbose=False)
        
        
        self.fab = fab_pt.FABAttack_PT(model, n_restarts=1, n_iter=100,
                                       eps=self.eps, seed=0,n_target_classes=9,
                                       verbose=False, device='cuda')
        
        self.square = square.SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.eps, norm='Linf',
                n_restarts=1, seed=0, verbose=False, device='cuda', resc_schedule=False)

        self.md = MD.MDAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps,loss_fn=self.loss_f)
        
        self.mdmt = MD.MDMTAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps)
        
        self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False,loss=self.loss_f)
        
        if self.attack_type == 'PGD':
            self.attacks_to_run = [self.pgd]

        elif self.attack_type == 'MTPGD':
            self.attacks_to_run = [self.mtpgd]
      
        elif self.attack_type == 'SFMT':
            self.attacks_to_run = [self.sfmt]
      
        elif self.attack_type == 'APGD':
            self.attacks_to_run = [self.apgd]
            
        elif self.attack_type == 'APGDT':
            self.attacks_to_run = []
            for i in range(1,10):
                self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps, 
                                                            num_classes=self.num_classes, seed=0, eot_iter=1, rho=.75, verbose=False,loss = self.loss_f, target=i)
                self.attacks_to_run.append(self.apgdt)
            print(len(self.attacks_to_run))
            #self.attacks_to_run = [self.apgdt]
            
        elif self.attack_type == 'MD':
            self.attacks_to_run = [self.md]
        
        elif self.attack_type == 'MDMT':
            self.attacks_to_run = [self.mdmt]
        
        elif self.attack_type == 'FAB':
            self.attacks_to_run = [self.fab]
        
        elif self.attack_type == 'Square':
            self.attacks_to_run = [self.square]
            
        elif self.attack_type == 'Test':
            self.pgd1 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type='CE')
            self.pgd2 = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=100,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type='MIFPE')
            self.attacks_to_run = [self.pgd1,self.pgd2]
            
        elif self.attack_type == 'MINE':
            self.md = self.md = MD.MDAttack(self.model,num_classes=self.num_classes,num_steps=100,num_random_starts=self.num_restarts,epsilon=self.eps,loss_fn="p_margin")
            
            self.attacks_to_run = [self.md]
            for i in range(1,10):
                self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps, 
                                                            num_classes=self.num_classes, seed=0, eot_iter=1, rho=.75, verbose=False,loss = 'Dlr', target=i)
                self.attacks_to_run.append(self.apgdt)
            print(len(self.attacks_to_run))
            
        elif self.attack_type == 'MINE2':
            self.md = self.md = MD.MDAttack(self.model,num_classes=self.num_classes,num_steps=100,num_random_starts=self.num_restarts,epsilon=self.eps,loss_fn="p_margin")
            self.apgdt = autopgd_pt.APGDAttack_targeted(self.model, n_iter=50, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False,loss='Dlr')
            self.attacks_to_run = [self.md,self.apgdt]
    
    def evaluate(self):
        adv_sum = 0
        clean_sum = 0
        total = self.y_test.shape[0]
        adv_images = self.x_test.clone()
        indexs = torch.tensor([i for i in range(total)],dtype=torch.int32)
        
        for i, a in enumerate(self.attacks_to_run):
            print(f"——————attack{i+1}——————")
            start = time.time()
            adv_indexs = torch.tensor([],dtype=torch.int32)
                
            for batch in tqdm(range(0,len(indexs),self.bs)):
                index = indexs[batch:(batch+self.bs)]
                images = self.x_test[index].to(device)
                labels = self.y_test[index].to(device)
                
                with torch.no_grad():
                    clean_logits = self.model(images)
                    if isinstance(clean_logits, list):
                        clean_logits = clean_logits[-1]
                    clean_pred = clean_logits.data.max(1)[1].detach()
                        
                accs = clean_pred == labels
                if i==0:
                    clean_sum += accs.float().sum()
                    #print(clean_sum)
                pre_acc = accs.float().mean()
                cor_indexs = accs.nonzero().squeeze()
                x_adv,adv_acc = a.perturb(images[cor_indexs], labels[cor_indexs])
                accs[cor_indexs] = adv_acc
                #pdb.set_trace()
                adv_indexs = torch.cat((adv_indexs,index[accs.cpu()]),dim=0)
                adv_images[index[cor_indexs.cpu()]] = x_adv.detach().cpu()
                
                print(f"pre_acc:{pre_acc*100:.2f}, adv_acc:{accs.float().mean()*100:.2f}")
            indexs = adv_indexs
            #pdb.set_trace()
            end = time.time()
            # Log
            if self.verbose:
                print(f"————after attack{i+1}————adv_acc:{len(indexs)/total*100:.2f}")
                print(f"————after attack{i+1}————time:{end-start}")
        #pdb.set_trace()
        #adv_indexs = [str(i) for i in range(len(adv_index)) if adv_index[i]]
        return (clean_sum/total).item(), len(indexs)/total,adv_images
