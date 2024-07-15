import timm
import torch
from torchvision import transforms
from utils import transform_and_stack

model_list = [
    'vgg16',
    'resnet101',
    'efficientnet_b3.ra2_in1k',
    'convnext_base',
    'vit_base_patch16_224',
    'vit_base_patch16_224.dino',
    'beit_base_patch16_224.in22k_ft_in22k_in1k',
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k',

    'inception_v3.tv_in1k',
    'resnet50',
    'deit3_base_patch16_224',
    'densenet121',
    'resnet18',

    'vit_base_patch14_reg4_dinov2.lvd142m',
]

no_head_model_list = [
    'vit_base_patch16_224.dino',
    'vit_base_patch14_reg4_dinov2.lvd142m',
]

def create_model_by_rank(rank, num_classes) -> torch.nn.Module:
    model = get_normalized_model(model_list[rank], num_classes)
    return model

class NModel(torch.nn.Module):
    def __init__(self, origin_model):
        super(NModel, self).__init__()
        self.origin_model = origin_model
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    
    def forward_features(self, x):
        if type(x) is list:
            x = transform_and_stack(x)
        x = self.normalize(x)
        return self.origin_model.forward_features(x)
    
    def forward(self, x):
        x = self.normalize(x)
        return self.origin_model.forward(x)

def get_normalized_model(origin_model, num_classes):
    if origin_model in no_head_model_list:
        num_classes = 0
    origin_model = timm.create_model(origin_model, pretrained=False, num_classes=num_classes, checkpoint_path=f'weights/{origin_model}.pth')
    return NModel(origin_model)