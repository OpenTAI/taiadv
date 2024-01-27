import models
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DefenseSATvgg(torch.nn.Module):
    destination = 'checkpoints/DefenseSATvgg/'

    def __init__(self):
        super(DefenseSATvgg, self).__init__()
        filename = 'checkpoints/N-vgg19_best.pth'
        checkpoint = torch.load(filename)['model_state_dict']

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        checkpoint = {strip_data_parallel(k): v for k, v in checkpoint.items()}
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        # Load Weights
        self.base_model = models.vgg.VGG(cfg=cfg)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)
