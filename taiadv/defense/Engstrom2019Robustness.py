import torch
from robustbench.utils import load_model

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Engstrom2019Robustness(torch.nn.Module):
    def __init__(self):
        super(Engstrom2019Robustness, self).__init__()
        self.base_model = load_model(model_name='Engstrom2019Robustness', model_dir='checkpoints',
                                     dataset='imagenet', threat_model='Linf')
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)
