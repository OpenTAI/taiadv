import os
from mmdet.apis import DetInferencer
from easydict import EasyDict
import glob
import argparse
import torch

def do_detection_evaluation(cfg):
    # get image file names
    torch.cuda.empty_cache()
    image_files = os.listdir(cfg.clean_image_path)
    image_files.sort()
    if cfg.num_images != 0:
        image_files = image_files[:cfg.num_images]
    clean_image_paths = [os.path.join(cfg.clean_image_path, image_file) for image_file in image_files]
    adv_image_paths = [os.path.join(cfg.adv_image_path, image_file) for image_file in image_files]
    inferencer = DetInferencer(cfg.model_name, cfg.model_path, device='cuda')
    ret_clean = inferencer(clean_image_paths, out_dir='outputs/', batch_size=cfg.batch_size)
    ret_adv = inferencer(adv_image_paths, out_dir='outputs/', batch_size=cfg.batch_size)
    ground_truth = []
    adv_ground = []
    for i in range(len(ret_clean['predictions'])):
        if ret_clean['predictions'][i]['labels'] and ret_adv['predictions'][i]['labels']:
            ground_truth.append((ret_clean['predictions'][i]['labels'][0], ret_clean['predictions'][i]['bboxes'][0]))
            adv_ground.append(ret_adv['predictions'][i])

    # calculate relative robustness
    def calc_IoU(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = w1 * h1
        boxBArea = w2 * h2
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    relative_robutsness = []
    for i in range(len(adv_ground)):
        cur_pred = adv_ground[i]
        for j in range(len(cur_pred['labels'])):
            if cur_pred['labels'][j] == ground_truth[i][0]:
                rr = calc_IoU(ground_truth[i][1], cur_pred['bboxes'][j])
                relative_robutsness.append(rr)
                break
    # print(f'{relative_robutsness=}')
    relative_robutsness = sum(relative_robutsness) / len(relative_robutsness)
    # print(f'average {relative_robutsness=}')
    with open(cfg.output, 'a') as f:
        f.write(args.model_name + ':' + str(relative_robutsness) + '\n')

def main(args):
    cfg = EasyDict(
        model_name = args.model_name,
        model_path = args.model_path,
        clean_image_path = args.clean_path,
        adv_image_path = args.adv_path,
        num_images = 1000,
        batch_size = 16,
        output = args.output
    )
    do_detection_evaluation(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="use model")
    parser.add_argument("--model_name", type=str, required=True, help="model_name")
    parser.add_argument("--model_path", type=str, required=True, help="model_path")
    parser.add_argument("--output", type=str, required=True, help="output file path")
    parser.add_argument("--clean_path", type=str, required=True, help="clean dataset path")
    parser.add_argument("--adv_path", type=str, required=True, help="adv dataset path")
    args = parser.parse_args()
    main(args)
