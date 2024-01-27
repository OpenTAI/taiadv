import models
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DefenseSATdensenet(torch.nn.Module):
    destination = 'checkpoints/DefenseSATdensenet/'

    def __init__(self):
        super(DefenseSATdensenet, self).__init__()
        filename = 'checkpoints/N-dense121_best.pth'
        checkpoint = torch.load(filename)['model_state_dict']

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        checkpoint = {strip_data_parallel(k): v for k, v in checkpoint.items()}

        # Load Weights
        self.base_model = models.densenet.RobustDenseNet(depth_configs=[6, 12, 24, 16],
                                                         channel_configs=[64, 128, 256, 512],
                                                         growth_rate=32)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)
