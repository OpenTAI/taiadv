import os
import torch
import timm
import torchvision
from torchvision import transforms
import torch.nn as nn
from easydict import EasyDict
import argparse
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file, root, transform=None, number=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if number == None:
            self.data_frame = pd.read_csv(csv_file)
        else:
            self.data_frame = pd.read_csv(csv_file, nrows=number)
        self.root_dir = root
        self.transform = transform
        self.number = number

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        if image.mode == 'L':
            image = image.convert('RGB')

        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def do_class_eval(cfg):
    output_file = cfg.output
    model_name = cfg.model_name
    print(f'模型{model_name}测试开始')
    adv_test_dataset = CustomDataset(csv_file=f'./groudtruth/plabel_{model_name}_cc1m.csv',root=cfg.data_path,transform=cfg.transform)
    test_loader = torch.utils.data.DataLoader(
    adv_test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=4)
    model_url = cfg.model_path
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(model_url)
    model.load_state_dict(state_dict)
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    correct = 0
    total = 0
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_total += loss.item()

    accuracy = 100 * correct / total
    loss_avg = loss_total / len(test_loader)
    with open(output_file, "a") as file:
        data = f'Model_name:{model_name} Accuracy:{accuracy:.2f}% loss: {loss_avg:.4f}'
        file.write(data + "\n")
    print(f'模型{model_name}测试结束')
    print('--------------------------------')

def main(args):
    cfg = EasyDict(
        model_name = args.model_name,
        model_path = args.model_path,
        data_path = args.data_path,
        output = args.output,
        batch_size = 16,
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
        ]),
    )
    do_class_eval(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="use model")
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--output", type=str, required=True, help="output file path")
    parser.add_argument("--data_path", type=str, required=True, help="dataset path")
    # parser.add_argument("--adv_path", type=str, required=True, help="adv dataset path")
    args = parser.parse_args()
    main(args)