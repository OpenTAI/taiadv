import os
from mmdet.apis import DetInferencer
from easydict import EasyDict
import argparse


def do_inst_seg_eval(cfg):
    # get image file names
    image_files = os.listdir(cfg.clean_image_path)
    image_files.sort()
    if cfg.num_images != 0:
        image_files = image_files[:cfg.num_images]
    clean_image_paths = [os.path.join(cfg.clean_image_path, image_file) for image_file in image_files]
    adv_image_paths = [os.path.join(cfg.adv_image_path, image_file) for image_file in image_files]

    inferencer = DetInferencer(cfg.model_name, cfg.model_path, device='cuda')

    # get ground truth. format: [(label, mask), ...]
    ret_clean = inferencer(clean_image_paths, out_dir='outputs/', batch_size=cfg.batch_size, return_datasamples=True)
    ret_adv = inferencer(adv_image_paths, out_dir='outputs/', batch_size=cfg.batch_size, return_datasamples=True)
    ground_truth = []
    adv_ground = []
    for i in range(len(ret_clean['predictions'])):
        if len(ret_clean['predictions'][i].pred_instances.labels) > 0 and len(ret_adv['predictions'][i].pred_instances.labels) > 0:
            ground_truth.append((ret_clean['predictions'][i].pred_instances.labels[0], ret_clean['predictions'][i].pred_instances.masks[0]))
            adv_ground.append(ret_adv['predictions'][i])

    # ground_truth = [(ret['predictions'][i].pred_instances.labels[0],ret['predictions'][i].pred_instances.masks[0]) for i in range(len(ret['predictions']))]
    
    # calculate relative robustness
    def calc_IoU(mask1, mask2):
        intersection = mask1 & mask2
        union = mask1 | mask2
        iou = intersection.sum() / union.sum()
        return iou.item()
    relative_robutsness = []
    # ret = inferencer(adv_image_paths, out_dir='outputs/', batch_size=cfg.batch_size, return_datasamples=True)
    for i in range(len(adv_ground)):
        cur_pred = adv_ground[i]
        for j in range(len(cur_pred.pred_instances.labels)):
            if cur_pred.pred_instances.labels[j] == ground_truth[i][0]:
                rr = calc_IoU(ground_truth[i][1], cur_pred.pred_instances.masks[j])
                relative_robutsness.append(rr)
                break
    # print(f'{relative_robutsness=}')
    relative_robutsness = sum(relative_robutsness) / len(relative_robutsness)
    with open(cfg.output, 'a') as f:
        f.write(args.model_name + ':' + str(relative_robutsness) + '\n')
    print(f'average {relative_robutsness=}')


def main(args):
    cfg = EasyDict(
        model_name = args.model_name,
        model_path = args.model_path,
        clean_image_path = args.clean_path,
        adv_image_path = args.adv_path,
        output = args.output,
        num_images = 1000,
        batch_size = 16,
    )
    do_inst_seg_eval(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="use model")
    parser.add_argument("--model_name", type=str, required=True, help="model_name")
    parser.add_argument("--model_path", type=str, required=True, help="model_path")
    parser.add_argument("--output", type=str, required=True, help="output file path")
    parser.add_argument("--clean_path", type=str, required=True, help="clean dataset path")
    parser.add_argument("--adv_path", type=str, required=True, help="adv dataset path")
    args = parser.parse_args()
    main(args)
