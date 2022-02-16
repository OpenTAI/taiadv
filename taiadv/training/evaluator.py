import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from . import util

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Evaluator():

    def __init__(self, criterion, data_loader, logger):
        self.loader = data_loader
        self.logger = logger
        self.criterion = criterion
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def eval(self, model, exp_stats={}):
        self._reset_stats()
        model.eval()
        for i, (images, labels) in enumerate(self.loader):
            self.eval_batch(images=images, labels=labels, model=model)
        payload = 'Val Loss: %.2f' % self.loss_meters.avg
        self.logger.info('\033[33m' + payload + '\033[0m')
        exp_stats['val_loss'] = self.loss_meters.avg
        payload = 'Val Acc: %.4f' % (self.acc_meters.avg * 100)
        self.logger.info('\033[33m' + payload + '\033[0m')
        exp_stats['val_acc'] = self.acc_meters.avg
        return exp_stats

    def eval_batch(self, images, labels, model):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        pred = logits.data.max(1)[1].detach()
        loss = loss.item()
        correct = (pred == labels.data).float().sum().item()
        self.loss_meters.update(loss, images.shape[0])
        acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        self.acc_meters.update(acc.item(), labels.shape[0])
        self.acc5_meters.update(acc5.item(), labels.shape[0])
        return correct, loss

    def adv_whitebox(self,
                     model,
                     epsilon=8 / 255,
                     num_steps=20,
                     step_size=0.8 / 255,
                     attacks='PGD',
                     exp_stats={}):
        model.eval()
        clean_correct, clean_loss, total = 0, 0, 0
        adv_correct, adv_loss, stable = 0, 0, 0

        start = time.time()
        for i, (images, labels) in enumerate(self.loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            correct, loss = self.eval_batch(images, labels, model)
            if attacks == 'PGD':
                rs = self._pgd_whitebox(
                    model,
                    images,
                    labels,
                    epsilon=epsilon,
                    num_steps=num_steps,
                    step_size=step_size)
            else:
                raise ('Unknown Attack')
            acc, acc_pgd, attack_loss, stable_count, _, = rs
            batch_size = images.shape[0]
            total += batch_size
            adv_correct += acc_pgd
            adv_loss += attack_loss * batch_size
            clean_loss += loss * batch_size
            clean_correct += correct
            stable += stable_count

        end = time.time()
        time_used = end - start

        rs = clean_loss / total, clean_correct, clean_correct / total * 100
        adv_rs = adv_loss / total, adv_correct, adv_correct / total * 100
        adv_stats = stable, stable / total * 100
        payload = 'Eval Loss: %.4f Eval Correct: %d Eval Acc: %.4f' % rs
        self.logger.info('\033[33m' + payload + '\033[0m')
        payload = 'Adv Loss: %.4f Adv Correct: %d Adv Acc: %.4f' % adv_rs
        self.logger.info('\033[33m' + payload + '\033[0m')
        payload = 'Adv Stable Count: %d Adv Stable: %.4f' % adv_stats
        self.logger.info('\033[33m' + payload + '\033[0m')
        payload = 'Time Cost: %.4f' % time_used
        self.logger.info('\033[33m' + payload + '\033[0m')
        exp_stats['clean_val_acc'] = clean_correct / total * 100
        exp_stats['clean_val_loss'] = clean_loss / total
        exp_stats['adv_val_acc'] = adv_correct / total * 100
        exp_stats['adv_val_loss'] = adv_loss / total
        exp_stats['adv_val_stable'] = stable
        exp_stats['adv_val_stable_precentage'] = stable / total * 100
        return exp_stats

    def _pgd_whitebox(self,
                      model,
                      X,
                      y,
                      random_start=True,
                      epsilon=0.031,
                      num_steps=20,
                      step_size=0.003):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(
                -epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = Variable(X_pgd.data, requires_grad=False)
        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum().item()
        stable = (predict_pgd.data == predict_clean.data).float().sum().item()
        return acc.item(), acc_pgd, loss.item(), stable, X_pgd
