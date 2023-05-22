import torch
import torch.nn.functional as F
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DenoisingAutoEncoder_1():
    def __init__(self, img_shape=(1,28,28)):
        self.img_shape = img_shape
    
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.AvgPool2d(2),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3,padding=1),
                nn.Sigmoid()).to(device)
    
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        optimizer = torch.optim.Adam(self.model.parameters())
        log_interval = 100
        for epoch in range(num_epochs):
            self.model.train()
            for batch_i,(data_train,data_label) in enumerate(data):
                noise = v_noise * torch.randn_like(data_train)
                noisy_data = torch.clamp(data_train+noise, min=min, max=max)
                data_train = torch.autograd.Variable(data_train).to(device)
                noisy_data = torch.autograd.Variable(noisy_data).to(device)
                optimizer.zero_grad()
                output = self.model(noisy_data)
                loss = F.mse_loss(output, data_train)
                loss.backward()
                optimizer.step()
                if batch_i % log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_i}/{len(data)} \t Loss: {loss.item()}]')

        if if_save:
            torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))


class DenoisingAutoEncoder_2():
    def __init__(self, img_shape = (1,28,28)):
        self.img_shape = img_shape
    
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1),
                nn.Sigmoid()).to(device)
        
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        optimizer = torch.optim.Adam(self.model.parameters())
        log_interval = 100
        for epoch in range(num_epochs):
            self.model.train()
            for batch_i,(data_train,data_label) in enumerate(data):
                noise = v_noise * torch.randn_like(data_train)
                noisy_data = torch.clamp(data_train+noise, min=min, max=max) 
                data_train = torch.autograd.Variable(data_train).to(device)
                noisy_data = torch.autograd.Variable(noisy_data).to(device)

                optimizer.zero_grad()
                output = self.model(noisy_data)
                loss = F.mse_loss(output,data_train)
                loss.backward()
                optimizer.step()
                if batch_i % log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_i}/{len(data)} \t Loss: {loss.item()}]')

        if if_save:
            torch.save(self.model.state_dict(), save_path)
        
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))
