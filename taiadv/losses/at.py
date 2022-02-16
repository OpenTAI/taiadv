import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StandardAT(nn.Module):

    def __init__(self,
                 step_size=0.007,
                 epsilon=0.031,
                 perturb_steps=10,
                 distance='l_inf'):
        """Implementation of standard adversarial training (SAT) based on
        "Towards Deep Learning Models Resistant to Adversarial Attacks" in ICLR
        2018."""

        super(StandardAT, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance

    def forward(self, model, images, labels):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # generate adversarial example
        x_adv = images.detach() + self.step_size * \
            torch.randn(images.shape).to(images.device)
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = F.cross_entropy(model(x_adv), labels)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                grad_sign = torch.sign(grad.detach())
                x_adv = x_adv.detach() + self.step_size * grad_sign
                x_adv = torch.min(
                    torch.max(x_adv, images - self.epsilon),
                    images + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            # only support L_inf for now.
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = Variable(x_adv, requires_grad=False)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        model.zero_grad()

        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)

        return logits, loss
