import torch
from easydict import EasyDict
import csv
import os
import timm
from torchvision import transforms
import argparse
from PIL import Image
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_files = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = self.img_files[index]
        img_path = os.path.join(self.data_dir, img_name)
        x = Image.open(img_path)
        if x.mode == 'L':
            x = x.convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, img_name

def generate_psudo_label(model, test_loader, file_name):
    cfg = EasyDict(
        target_file_name = file_name, 
        batch_size = 128,
        num_images = 1000,
    )
    with open(cfg.target_file_name, 'w') as file:
        writer = csv.writer(file)
        with torch.no_grad():
            for i,(x, img_names) in enumerate(test_loader):
                print('processing',i*cfg.batch_size, flush=True)
                x = x.cuda()
                outputs = model(x)
                _, yp = torch.max(outputs.data, 1)
                batch_info = [(img_names[i],yp[i].item()) for i in range(len(x))]
                writer.writerows(batch_info)


def do_generate(cfg):
    test_dataset = MyDataset(data_dir = cfg.data_path,transform=cfg.transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4
    )
    model_name = cfg.model_name
    print(f'--------{model_name}生成开始-----------')
    model_url = cfg.model_path
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(model_url)
    model.load_state_dict(state_dict)
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    file_name = f'./groudtruth/plabel_{model_name}_cc1m.csv'
    generate_psudo_label(model, test_loader, file_name)

def main(args):
    cfg = EasyDict(
        model_name = args.model_name,
        model_path = args.model_path,
        data_path = args.data_path,
        batch_size = 16,
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
        ]),
    )
    do_generate(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="use model")
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    # parser.add_argument("--output", type=str, required=True, help="output file path")
    parser.add_argument("--data_path", type=str, required=True, help="dataset path")
    # parser.add_argument("--adv_path", type=str, required=True, help="adv dataset path")
    args = parser.parse_args()
    main(args)