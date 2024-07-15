import torch
from torchvision import transforms, utils
import os

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
transform_aug = transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224)])

def transform_and_stack(x_list, aug=False):
    if aug:
        x_tensor = [transform_aug(x) for x in x_list]
    else:
        x_tensor = [transform(x) for x in x_list]
    x = torch.stack(x_tensor)
    return x

id_cnt = 0
def save_raw(adv_image, y, path = './'):
    global id_cnt
    print('save batch', id_cnt)
    torch.save(adv_image,os.path.join(path, f'{id_cnt}_image.pt'))
    torch.save(y,os.path.join(path, f'{id_cnt}_label.pt'))
    id_cnt+=1

def extract_images(raw_path, dest_path):
    dirs = os.listdir(raw_path)
    print(len(dirs))
    size = int(len(dirs)/2)
    for i in range(size):
        print(f'batch {i}')
        image_list = torch.load(os.path.join(raw_path, f'{i}_image.pt'))
        y_list = torch.load(os.path.join(raw_path, f'{i}_label.pt'))
        for image, y in zip(image_list, y_list):
            utils.save_image(image, os.path.join(dest_path, y))