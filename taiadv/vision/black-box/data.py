import os
import csv
from PIL import Image
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset as TDataset

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    if type(target[0]) == torch.Tensor:
        target = torch.LongTensor(target)
    return [data, target]

def get_data_loader(dataset_name, root_folder='../data', batch_size=128, aug = False, separate = False):
    testset = Dataset(root_folder).get_test_dataset(dataset_name, aug)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn= my_collate if separate else None)
    return testloader

num_classes_dict = {'imagenet':1000,'cars':196,'pets':37,'food':101,'DTD':47,'cifar10':10,'cifar100':100,'fgvc':100,'cub':200,'svhn':10,'stl10':10,}

class Dataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder


    def get_test_dataset(self, name, data_aug = False):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(os.path.join(self.root_folder,'cifar10'), train=False,
                                                              transform=self.Transform_with_aug() if data_aug == True else self.Transform_without_aug() if data_aug == False else data_aug),
                          'imagenet':lambda: datasets.ImageFolder(os.path.join(self.root_folder,'imagenet_2012','val'),
                                                               transform=self.Transform_with_aug() if data_aug == True else self.Transform_without_aug() if data_aug == False else data_aug),
                          'coco2017': lambda: SimpleImageFolder(os.path.join(self.root_folder,'coco2017','val2017'),
                                                transform=self.Transform_with_aug() if data_aug == True else self.Transform_without_aug() if data_aug == False else data_aug),
                          'market1501': lambda: SimpleImageFolder(os.path.join(self.root_folder,'Market-1501-v15.09.15','bounding_box_test'),
                                                transform=self.Transform_with_aug() if data_aug == True else self.Transform_without_aug() if data_aug == False else data_aug),
                          'ade20k': lambda: SimpleImageFolder(os.path.join(self.root_folder,'ADEChallengeData2016','images','validation'),
                                                transform=self.Transform_with_aug() if data_aug == True else self.Transform_without_aug() if data_aug == False else data_aug),
                          'cc1m': lambda: SimpleImageFolder(os.path.join(self.root_folder,'cc1m','images'),
                                                transform=self.Transform_with_aug() if data_aug == True else self.Transform_without_aug() if data_aug == False else data_aug),
                          'chexpert': lambda: RecursiveImageFolder(os.path.join(self.root_folder,'CheXpert-v1.0-small','valid'),
                                                transform=self.Transform_with_aug() if data_aug == True else self.Transform_without_aug() if data_aug == False else data_aug),
                          }
        dataset_fn = valid_datasets[name]
        return dataset_fn()
    
    def get_num_classes(self, name):
        return num_classes_dict[name]

    @staticmethod
    def Transform_with_aug():
        return transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor()])

    @staticmethod
    def Transform_without_aug():
        return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        # return transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

def get_adv_data_loader(dataset_name, root_folder='./adv_image'):
    origin_testset = Dataset('../data').get_test_dataset(dataset_name)
    testloader = DataLoader(origin_testset, batch_size=128, shuffle=False, num_workers=2)
    for i in range(0,391):
        # load the adv images
        adv_images = torch.load(f'{root_folder}/batch_{i}.pth')
        images, labels = next(iter(testloader))
        images = images.cuda()
        assert (adv_images - images).max() < 9/255
        yield adv_images, labels

class SimpleImageFolder(TDataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.dirs = os.listdir(root)
        
        # for market1501
        self.dirs = [dir for dir in self.dirs if dir != 'Thumbs.db']

        self.dirs.sort()
        self.len = len(self.dirs)

    def __getitem__(self, index: int):
        img_name = self.dirs[index]
        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # label位置 复用为文件名
        return img, img_name

    def __len__(self) -> int:
        return self.len

class RecursiveImageFolder(TDataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.full_paths, self.rela_paths = self.get_image_paths(root)
        self.len = len(self.full_paths)
                
    def get_image_paths(self, directory):
        full_paths = []
        rela_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                fullPath = os.path.join(root, file)
                full_paths.append(fullPath)
                sp = fullPath.split('/')
                newPath = '/'.join(sp[-3:])
                rela_paths.append(newPath)
                print(fullPath)
        return full_paths, rela_paths

    def __getitem__(self, index: int):
        img_name = self.rela_paths[index]
        img = Image.open(self.full_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # label位置 复用为文件名
        return img, img_name

    def __len__(self) -> int:
        return self.len


if __name__ == '__main__':
    dataset = Dataset('../data').get_test_dataset('coco2017')
    print(len(dataset))
    print(dataset[0][0].shape)
    # print(dataset[0][0].shape)