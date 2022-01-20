import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MART(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10,
                 distance='l_inf', beta=6.0):
        """
            Implementation of MART based on "Improving Adversarial Robustness
            Requires Revisiting Misclassified Examples" in ICLR 2020
            https://github.com/YisenWang/MART
        """

        super(MART, self).__init__()
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
                x_adv = torch.min(torch.max(x_adv, images - self.epsilon),
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

        batch_size = images.shape[0]
        logits = model(images)
        logits_adv = model(x_adv)

        adv_probs = F.softmax(logits_adv, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

        new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_adv = self.cross_entropy(logits_adv, labels) +\
            F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(
            nat_probs, 1, (labels.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(self.kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1)
            * (1.0000001 - true_probs))
        loss = loss_adv + float(self.beta) * loss_robust

        return logits_adv, loss
