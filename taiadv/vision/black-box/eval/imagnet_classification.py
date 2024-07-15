import os
import torch
import timm
import torchvision
from torchvision import transforms
import torch.nn as nn
from easydict import EasyDict
import argparse

def do_class_eval(cfg):
    clean_test_dataset = torchvision.datasets.ImageFolder(
        root = cfg.clean_image_path,
        transform = cfg.transform
    )
    adv_test_dataset = torchvision.datasets.ImageFolder(
        root = cfg.adv_image_path,
        transform = cfg.transform
    )
    test_loader = torch.utils.data.DataLoader(
        clean_test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4
    )
    adv_test_loader = torch.utils.data.DataLoader(
        adv_test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4
    )
    model_url = cfg.model_path
    model = timm.create_model(cfg.model_name, pretrained=False)
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

    accuracy_clean = 100 * correct / total
    loss_avg_clean = loss_total / len(test_loader)
    correct = 0
    total = 0
    loss_total = 0

    with torch.no_grad():
        for images, labels in adv_test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_total += loss.item()

    accuracy_adv = 100 * correct / total
    loss_avg_adv = loss_total / len(test_loader)
    with open(cfg.output, "a") as file:
        data = f'Model_name:{cfg.model_name} Accuracy_clean:{accuracy_clean:.2f}% Accuracy_adv:{accuracy_adv:.2f}% loss_clean: {loss_avg_clean:.4f} loss_adv: {loss_avg_adv:.4f}'
        file.write(data + "\n")
    print(f'模型{cfg.model_name}测试结束')
    print('--------------------------------')

def main(args):
    cfg = EasyDict(
        model_name = args.model_name,
        model_path = args.model_path,
        clean_image_path = args.clean_path,
        adv_image_path = args.adv_path,
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
    parser.add_argument("--clean_path", type=str, required=True, help="clean dataset path")
    parser.add_argument("--adv_path", type=str, required=True, help="adv dataset path")
    args = parser.parse_args()
    main(args)