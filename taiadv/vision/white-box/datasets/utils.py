from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageFolder
from .cc1m import CC1M

def _convert_to_rgb(image):
    return image.convert('RGB')

transform_options = {
    "CIFAR10": {
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "test_transform": [transforms.ToTensor()]},
    "SVHN": {
        "test_transform": [transforms.ToTensor()]},
    "MNIST": {
        "test_transform": [transforms.ToTensor()]},
    "ImageNet": {
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]},
    "CC1M":{
        "test_transform":[transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor()],
    }
}

transform_options["CIFAR10Noisy"] = transform_options["CIFAR10"]

dataset_options = {
    "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train= not is_test, download=False,
                transform=transform),
    "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train= not is_test, download=False,
                 transform=transform),
    "SVHN": lambda path, transform, is_test, kwargs:
        SVHN(root=path, split='test', download=False,
             transform=transform),
    "MNIST": lambda path, transform, is_test, kwargs:
        MNIST(root=path, train=not not is_test, download=False,
              transform=transform),
    "ImageNet": lambda path, transform, is_test, kwargs:
        ImageFolder(root=path+'val',transform=transform),
    "CC1M":lambda path, transform, is_test, kwargs:
        CC1M(root=path)
}
