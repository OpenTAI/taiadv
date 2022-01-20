import time
import torch
from . import util

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer():
    def __init__(self, criterion, data_loader, logger, log_frequency=25,
                 global_step=0):
        self.loader = data_loader
        self.logger = logger
        self.criterion = criterion
        self.log_frequency = log_frequency
        self.global_step = global_step
        self._reset_stats()

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train_batch(self, images, labels, model, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels)
        loss.backward()
        grad_norm = 0
        optimizer.step()

        acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        self.acc_meters.update(acc.item(), labels.shape[0])
        self.acc5_meters.update(acc5.item(), labels.shape[0])
        self.loss_meters.update(loss.item(), labels.shape[0])
        payload = {
            'acc': acc.item(),
            'acc_avg': self.acc_meters.avg,
            "loss": loss.item(),
            "loss_avg": self.loss_meters.avg,
            "lr": optimizer.param_groups[0]['lr'],
            "|gn|": grad_norm}
        return payload

    def train(self, epoch, model, optimizer, exp_stats={}):
        model.train()
        self._reset_stats()
        for i, data in enumerate(self.loader):
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        exp_stats['global_step'] = self.global_step
        exp_stats['lr'] = optimizer.param_groups[0]['lr']
        exp_stats['train_loss'] = self.loss_meters.avg
        exp_stats['train_acc'] = self.acc_meters.avg
        return exp_stats
