import time
import sys
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

torch.manual_seed(0)

def Dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
        1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def Margin_loss(logits, y, num_classes):
    device = logits.device
    logit_org = logits.gather(1, y.view(-1, 1))
    logit_target = logits.gather(1, (logits - torch.eye(num_classes)[y.to("cpu")].to(device) * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def Probability_Margin(logits, y, num_classes):
    device = logits.device
    logits = F.softmax(logits,dim=-1) #softmax
    logit_org = logits.gather(1, y.view(-1, 1))
    logit_target = logits.gather(1, (logits - torch.eye(num_classes)[y.to("cpu")].to(device) * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss
    
def MIFPE(logits,y):
    device = logits.device
    t = 1.0
    def get_output_scale(output):
        std_max_out = []
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [item / t for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list
    scale_output = get_output_scale(logits.clone().detach())
    logits = logits/scale_output
    loss = F.cross_entropy(logits, y)
    return loss

def Alt_Loss(logits,y,e,epochs,num_classes):
    if e<epochs/3:
        return F.cross_entropy(logits,y)
    elif e<epochs/3*2:
        return Dlr_loss(logits,y).sum()
    else:
        return Margin_loss(logits,y,num_classes)

class PGDAttack():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_restarts=1, v_min=0., v_max=1., num_classes=10,
                 random_start=False, loss_type='ce',decay_step='cos', use_odi=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.decay_step = decay_step
        self.use_odi = use_odi
        self.step_size = step_size

    def _get_rand_noise(self, X):
        eps = self.epsilon
        device = X.device
        return torch.FloatTensor(*X.shape).uniform_(-eps, eps).to(device)

    def perturb(self, x_in, y_in):
        model = self.model
        device = x_in.device
        epsilon = self.epsilon
        X_adv = x_in.detach().clone()
        step_size_begin = 2.0/255

        with torch.no_grad():
            clean_logits = model(x_in)
            if isinstance(clean_logits, list):
                clean_logits = clean_logits[-1]
            clean_pred = clean_logits.data.max(1)[1].detach()
            accs = clean_pred == y_in

        for _ in range(self.num_restarts):
            if self.random_start:
                random_noise = self._get_rand_noise(x_in[accs])
                X_adv[accs] = x_in[accs] + random_noise
            else:
                X_pgd = X_adv[accs]

            if self.use_odi:
                out = model(x_in)
                rv = torch.FloatTensor(*out.shape).uniform_(-1., 1.).to(device)

            cor_indexs = accs.nonzero()
            if cor_indexs.numel()>0:
                cor_indexs = cor_indexs.squeeze()
            else:
                break
            
            x_pgd = X_adv[cor_indexs]
            y = y_in[cor_indexs].clone()
            
            if len(x_pgd.shape) != len(x_in.shape):
                x_pgd = x_pgd.unsqueeze(0)
                y = y.unsqueeze(0)
            x_pgd = Variable(x_pgd,requires_grad=True)

            for i in range(self.num_steps):
                if self.decay_step == 'linear':
                    step_size = step_size_begin * (1 - i / self.num_steps)
                elif self.decay_step == 'cos':
                    step_size = step_size_begin * math.cos(i / self.num_steps * math.pi * 0.5)
                else:
                    step_size = self.step_size
                
                #optimizer.zero_grad()
                logit = model(x_pgd)
                if self.use_odi and i < 2:
                    loss = (logit * rv).sum()
                elif self.loss_type == 'ce':
                    loss = F.cross_entropy(logit,y)
                ####
                elif self.loss_type == 'cet':
                    logit_y = torch.eye(self.num_classes)[y.to("cpu")].to("cuda")*logit*1e8
                    target = (logit - logit_y).argmax(dim=-1)
                    loss = -F.cross_entropy(logit,target)
                elif self.loss_type == 'dlr':
                    loss = Dlr_loss(logit,y).sum()
                elif self.loss_type == 'mg':
                    loss = Margin_loss(logit,y,self.num_classes)
                elif self.loss_type == 'pm':
                    loss = Probability_Margin(logit,y,self.num_classes)
                elif self.loss_type == 'alt':
                    loss = Alt_Loss(logit,y,i,self.num_steps,self.num_classes)
                elif self.loss_type == 'mi':
                    loss = MIFPE(logit,y)
                elif self.loss_type == 'mi_t':
                    logit_y = torch.eye(self.num_classes)[y.to("cpu")].to("cuda")*logit*1e8
                    target = (logit - logit_y).argmax(dim=-1)
                    loss = -MIFPE(logit,target)
                else:
                    raise("error")
                loss.backward()
                
                acc = logit.max(1)[1].detach()==y
                accs[cor_indexs] = acc
                X_adv[cor_indexs] = x_pgd.detach()
                
                if self.use_odi and i < 2:
                    eta = epsilon * x_pgd.grad.data.sign()
                else:
                    eta = step_size * x_pgd.grad.data.sign()
                x_pgd = torch.clamp(x_pgd.data + eta, 0.,1.)
                
                eta = torch.clamp(x_pgd.data - x_in[cor_indexs], -epsilon, epsilon)
                x_pgd = torch.clamp(x_in[cor_indexs] + eta,0.,1.)
                x_pgd = Variable(x_pgd[acc],requires_grad=True)
                
                cor_indexs = accs.nonzero()
                if cor_indexs.numel()>0:
                    cor_indexs = cor_indexs.squeeze()
                else:
                    continue
                y = y_in[cor_indexs]
                X_adv[cor_indexs] = x_pgd.detach()
                    
                
            with torch.no_grad():
                logits = self.model(x_pgd)
                acc = logits.max(1)[1]==y
                accs[cor_indexs] = acc
                X_adv[cor_indexs] = x_pgd
        return X_adv,accs
