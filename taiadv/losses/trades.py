import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class TRADES(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10,
                 distance='l_inf', beta=6.0):
        """
            Implementation of TRADES based on "Theoretically Principled
            Trade-off between Robustness and Accuracy" in ICML 2019
            https://github.com/yaodongyu/TRADES
        """

        super(TRADES, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.kl = nn.KLDivLoss(reduction='sum')

    def forward(self, model, images, labels):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # generate adversarial example
        batch_size = len(images)
        logits = model(images)
        x_adv = images.detach() + 0.001 *\
            torch.randn(images.shape).to(images.device).detach()

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_kl = self.kl(F.log_softmax(model(x_adv), dim=1),
                                  F.softmax(logits, dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                grad_sign = torch.sign(grad.detach())
                x_adv = x_adv.detach() + self.step_size * grad_sign
                x_adv = torch.min(torch.max(x_adv, images - self.epsilon),
                                  images + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.distance == 'l_2':
            delta = 0.001 * torch.randn(images.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD(
                [delta], lr=self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = images + delta

                # optimize
                optimizer_delta.zero_grad()
                loss = (-1) * self.kl(F.log_softmax(model(adv), dim=1),
                                      F.softmax(logits, dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(
                        delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(images)
                delta.data.clamp_(0, 1).sub_(images)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_adv = Variable(images + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = Variable(x_adv, requires_grad=False)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        model.zero_grad()

        batch_size = images.shape[0]
        logits = model(images)
        loss_natural = self.cross_entropy(logits, labels)
        adv_logits = model(x_adv)
        loss_robust = (1.0 / batch_size) *\
            self.kl(F.log_softmax(adv_logits, dim=1),
                    F.softmax(logits, dim=1))

        loss = loss_natural + self.beta * loss_robust
        return adv_logits, loss
