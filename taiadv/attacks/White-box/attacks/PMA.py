import time
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .utils import adv_check_and_update, one_hot_tensor
import math

torch.manual_seed(0)

class PMA():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_random_starts=1, v_min=0., v_max=1., change_point=50,
                 first_step_size=16./255., seed=0, norm='Linf', num_classes=100,
                 use_odi=False, use_dlr=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_random_starts = num_random_starts
        self.v_min = v_min
        self.v_max = v_max
        self.change_point = num_steps/2
        self.first_step_size = first_step_size
        self.seed = seed
        self.norm = norm
        self.num_classes = num_classes
        self.use_odi = use_odi
        self.use_dlr = use_dlr
        self.loss_fn = loss_fn
        self.initial_step_size = 2.0 * epsilon

    def perturb(self, x_in, y_in):
        change_point = self.change_point
        X_adv = x_in.detach().clone()

        with torch.no_grad():
            logits = self.model(x_in)
        pred = logits.max(dim=1)[1]
        accs = pred==y_in
        for _ in range(max(self.num_random_starts, 1)):
            for r in range(2):
                r_noise = torch.FloatTensor(*x_in[accs].shape).uniform_(-self.epsilon, self.epsilon).cuda()
                X_adv[accs] = x_in[accs].data + r_noise
                
                cor_indexs = accs.nonzero().squeeze()
                x_pgd = Variable(X_adv[cor_indexs] + 0.,requires_grad=True)
                y = y_in[cor_indexs]
                
                for i in range(self.num_steps):
                    if x_pgd.shape[0]==0 or len(x_pgd.shape)!=4:
                        return X_adv,accs
                    with torch.enable_grad():
                        logits = self.model(x_pgd)
                        logits = F.softmax(logits,dim=-1)
                        
                        z_y = logits.gather(1, y.view(-1, 1))
                        
                        #pdb.set_trace()
                        
                        z_max = logits.gather(1, (logits - torch.eye(self.num_classes)[y.cpu()].to("cuda") * 9999).argmax(1, keepdim=True))

                        if i < 1:
                            loss_per_sample = z_y
                            loss = torch.mean(loss_per_sample)
                        elif i < change_point:
                            loss_per_sample = z_max if r else -z_y
                            loss = torch.mean(loss_per_sample)
                        else:
                            loss_per_sample = z_max - z_y
                            loss = torch.mean(loss_per_sample)
                        loss.backward()
                        
                        acc = logits.max(1)[1] == y
                        
                        accs[cor_indexs] = acc
                        X_adv[cor_indexs] = x_pgd.detach()

                    if self.use_odi and i < 2:
                        alpha = self.epsilon
                    elif i > change_point:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i-change_point - 1) / (self.num_steps-change_point) * np.pi))
                    else:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i - 1) / (self.num_steps-change_point) * np.pi))
                    eta = alpha * x_pgd.grad.data.sign()
                    x_pgd = x_pgd.detach() + eta.detach()
                    x_pgd = torch.min(torch.max(x_pgd, x_in[cor_indexs] - self.epsilon), x_in[cor_indexs] + self.epsilon)
                    x_pgd = torch.clamp(x_pgd, self.v_min, self.v_max)
                    x_pgd = Variable(x_pgd[acc],requires_grad=True)
                    
                    cor_indexs = accs.nonzero().squeeze()
                    y = y_in[cor_indexs]
                    if cor_indexs.numel()==0:
                        return X_adv,accs
                    
                with torch.no_grad():
                    logits = self.model(x_pgd)
                acc = logits.max(1)[1]==y
                accs[cor_indexs] = acc
                X_adv[cor_indexs] = x_pgd
                    
        return X_adv, accs

