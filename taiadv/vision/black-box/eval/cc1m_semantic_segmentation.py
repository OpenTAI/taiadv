import os
import glob
import argparse
import torch
from mmseg.apis import init_model, inference_model
from mmseg.evaluation import metrics
from easydict import EasyDict
import argparse
from tqdm import tqdm


def cal_iou(clean_data_sample, adv_data_sample, num_classes):
    device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_label = adv_data_sample.pred_sem_seg.data.squeeze()
    true_label = clean_data_sample.pred_sem_seg.data.squeeze()
    pred_one_hot = torch.zeros(num_classes, pred_label.size(0), pred_label.size(1))
    true_one_hot = torch.zeros(num_classes, true_label.size(0), true_label.size(1))
    pred_one_hot = pred_one_hot.to(device = device1)
    true_one_hot = true_one_hot.to(device = device1)
    pred_one_hot = pred_one_hot.scatter_(0, pred_label.unsqueeze(0), 1)
    true_one_hot = true_one_hot.scatter_(0, true_label.unsqueeze(0), 1)
    intersection = (pred_one_hot * true_one_hot).sum(dim=(1, 2))
    union = (pred_one_hot + true_one_hot - (pred_one_hot * true_one_hot)).sum(dim=(1, 2))
    epsilon = 1e-6
    # iou = intersection / union
    iou = intersection / (union + epsilon)
    non_zero_values = iou[iou != 0]
    avg_non_zero_iou = non_zero_values.mean()
    return avg_non_zero_iou


def evl_robust(cfg):
    image_files = os.listdir(cfg.clean_image_path)
    image_files.sort()
    if cfg.num_images != 0:
        image_files = image_files[:cfg.num_images]
    clean_image_paths = [os.path.join(cfg.clean_image_path, image_file) for image_file in image_files]
    adv_image_paths = [os.path.join(cfg.adv_image_path, image_file) for image_file in image_files]
    total_images = len(clean_image_paths)
    model = init_model(cfg.model_config, cfg.model_path)
    relative_robustes = []
    batch_size = cfg.batch_size
    for batch_start in tqdm(range(0, total_images, batch_size), desc='Processing batches'):
        batch_end = batch_start + batch_size
        if batch_end > total_images:
            batch_end = total_images
            batch_size = total_images - batch_start
        clean_batch = clean_image_paths[batch_start:batch_end]
        adv_batch = adv_image_paths[batch_start:batch_end]
        clean_result = inference_model(model, clean_batch)
        adv_result = inference_model(model, adv_batch)
        for i in range(batch_size):
            iou = cal_iou(clean_result[i], adv_result[i], cfg.num_classes)
            if torch.isnan(iou).all():
                continue
            relative_robustes.append(iou)
    relative_robust = sum(relative_robustes) / len(relative_robustes)
    return relative_robust

def main(args):
    cfg = EasyDict(
        model_config = args.model_config,
        model_path = args.model_path,
        clean_image_path = args.clean_path,
        adv_image_path = args.adv_path,
        num_images = 50000,
        batch_size = 32,
        num_classes = 150,
        output = args.output,
    )
    relative_robust = evl_robust(cfg)
    with open(cfg.output, 'a') as f:
        f.write(args.model_name + ':' + str(relative_robust) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="use model")
    parser.add_argument("--model_name", type=str, required=True, help="model_name")
    parser.add_argument("--model_path", type=str, required=True, help="model_path")
    parser.add_argument("--model_config", type=str, required=True, help="model_config")
    parser.add_argument("--output", type=str, required=True, help="output file path")
    parser.add_argument("--clean_path", type=str, required=True, help="clean dataset path")
    parser.add_argument("--adv_path", type=str, required=True, help="adv dataset path")
    args = parser.parse_args()
    main(args)