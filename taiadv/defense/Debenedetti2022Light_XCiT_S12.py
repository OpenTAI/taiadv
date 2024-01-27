import torch
from robustbench.utils import load_model

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Debenedetti2022Light_XCiT_S12(torch.nn.Module):
    def __init__(self):
        super(Debenedetti2022Light_XCiT_S12, self).__init__()
        self.base_model = load_model(model_name='Debenedetti2022Light_XCiT-S12', model_dir='checkpoints',
                                     dataset='imagenet', threat_model='Linf')
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)
