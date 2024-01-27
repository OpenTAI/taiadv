import models
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DefenseSATdarts(torch.nn.Module):
    destination = 'checkpoints/DefenseSATdarts/'

    def __init__(self):
        super(DefenseSATdarts, self).__init__()
        filename = 'checkpoints/N-DARTS.pth'
        checkpoint = torch.load(filename)['model_state_dict']

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        checkpoint = {strip_data_parallel(k): v for k, v in checkpoint.items()}

        # Load Weights
        self.base_model = models.darts.NetworkCIFAR(num_classes=10, genotype='DARTS_ORIGINAL', layers=11,
                                                    auxiliary=True, aux_weights=0.4, drop_path_prob=0.0,
                                                    channel_configs=[108, 72, 144, 288])
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)
