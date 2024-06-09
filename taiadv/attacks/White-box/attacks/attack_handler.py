import torch
import numpy as np
from . import PGD,APGD,APGDT,PMA,FAB
from tqdm.auto import tqdm
import time

class Attacker():
    def __init__(self, model, attack_type = 'PGD_attack', eps=8./255, random_start=False,
                 noise = 'Uniform', num_restarts=1, step_size=2./255, bs = 32,
                 num_steps=100, loss_f='CE', use_odi=False,num_classes=10,
                 verbose=True, logger=None):
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
        self.bs = bs
        self.logger = logger
        self.verbose = verbose
        
        self.pgd = PGD.PGDAttack(self.model,epsilon=self.eps,num_steps=self.num_steps,step_size=self.step_size,num_restarts=self.num_restarts,
                       num_classes=self.num_classes,random_start=self.random_start,loss_type=self.loss_f)
        
        self.apgd = APGD.APGDAttack(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, loss=self.loss_f, eot_iter=1, rho=.75, verbose=False)
        
        self.fab = FAB.FABAttack_PT(model, n_restarts=1, n_iter=100,
                                       eps=self.eps, seed=0,n_target_classes=9,
                                       verbose=False, device='cuda')

        self.pma = PMA.MDAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps,loss_fn='pm',use_odi=self.use_odi)
        
        self.md = PMA.MDAttack(self.model,num_classes=self.num_classes,num_steps=self.num_steps,num_random_starts=self.num_restarts,epsilon=self.eps,loss_fn='mg',use_odi=self.use_odi)
        
        self.apgdt = APGDT.APGDAttack_targeted(self.model, n_iter=self.num_steps, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False,loss=self.loss_f)
        
        if self.attack_type == 'PGD':
            self.attacks_to_run = [self.pgd]
      
        elif self.attack_type == 'APGD':
            self.attacks_to_run = [self.apgd]
            
        elif self.attack_type == 'APGDT':
            self.attacks_to_run = [self.apgdt]
            
        elif self.attack_type == 'MD':
            self.attacks_to_run = [self.md]
        
        elif self.attack_type == 'FAB':
            self.attacks_to_run = [self.fab]
            
        elif self.attack_type == 'PMA':
            self.attacks_to_run = [self.pma]
            
        elif self.attack_type == 'PMA+':
            self.pma = PMA.MDAttack(self.model,num_classes=self.num_classes,num_steps=100,num_random_starts=self.num_restarts,epsilon=self.eps,loss_fn="pm")
            self.apgdt = APGDT.APGDAttack_targeted(self.model, n_iter=100, norm='Linf', n_restarts=self.num_restarts, eps=self.eps,
                 seed=0, eot_iter=1, rho=.75, verbose=False,loss='Dlr')
            self.attacks_to_run = [self.pma,self.apgdt]
