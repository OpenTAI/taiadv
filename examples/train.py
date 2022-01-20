import argparse
import mlconfig
import torch
import random
import numpy as np
import time
import os
import taiadv
import taiadv.losses
import taiadv.models
from taiadv.training.exp_mgmt import ExperimentManager
from taiadv.training.trainer import Trainer
from taiadv.training.evaluator import Evaluator
from taiadv.datasets.data import DatasetGenerator
print(ExperimentManager)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Register mlconfig
mlconfig.register(DatasetGenerator)
mlconfig.register(taiadv.losses.StandardAT)
mlconfig.register(taiadv.losses.TRADES)
mlconfig.register(taiadv.losses.MART)
mlconfig.register(taiadv.models.rwrn.RobustWideResNet)
mlconfig.register(taiadv.models.wrn.WideResNet)
mlconfig.register(taiadv.models.rn.ResNet18)
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
mlconfig.register(torch.nn.CrossEntropyLoss)


parser = argparse.ArgumentParser(description='OpenTAI Adv')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options
parser.add_argument('--exp_name', type=str, default='wrn34x10_sat')
parser.add_argument('--exp_path', type=str, default='experiments/cifar10')
parser.add_argument('--exp_config', default='configs/', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_best_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
# Adv Attack Options
parser.add_argument('--eps', default=8, type=float)
parser.add_argument('--num_steps', default=20, type=int)
parser.add_argument('--step_size', default=0.8, type=float)
parser.add_argument('--attack_choice', default='PGD')


def train(args, exp, model, optimizer, scheduler, criterion, loader,
          start_epoch):
    train_loader, eval_loader = loader
    logger = exp.logger
    best_acc = 0

    if args.load_model or args.load_best_model:
        prev_stats = exp.load_epoch_stats()
    else:
        prev_stats = None

    if prev_stats is not None:
        step = prev_stats['global_step']
        best_acc = prev_stats['best_acc']
    else:
        step = 0
    trainer = Trainer(criterion=criterion, data_loader=train_loader,
                      logger=logger, global_step=step,
                      log_frequency=config.log_frequency)
    evaluator = Evaluator(criterion=criterion, data_loader=eval_loader,
                          logger=logger)
    for epoch in range(start_epoch, config.epochs):
        exp_stats = {}

        # train
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.train()
        exp_stats = trainer.train(epoch=epoch, model=model,
                                  optimizer=optimizer, exp_stats=exp_stats)
        scheduler.step()

        # adversarial evaluation
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        model.eval()
        exp_stats = evaluator.adv_whitebox(model, epsilon=args.eps,
                                           num_steps=args.num_steps,
                                           step_size=args.step_size,
                                           attacks=args.attack_choice,
                                           exp_stats=exp_stats)
        acc = exp_stats['adv_eval_acc']

        if acc > best_acc:
            is_best = True
            best_acc = acc
        else:
            is_best = False

        payload = 'Best Acc: %.2f' % best_acc
        logger.info('\033[33m'+payload+'\033[0m')
        exp_stats['best_acc'] = best_acc
        exp_stats['epoch'] = epoch

        # update exp stats
        exp.save_epoch_stats(epoch=epoch, exp_stats=exp_stats)

        # save model
        exp.save_state(model, 'model_state_dict')
        exp.save_state(optimizer, 'optimizer_state_dict')
        exp.save_state(scheduler, 'scheduler_state_dict')

        if is_best:
            exp.save_state(model, 'model_state_dict_best')
            exp.save_state(optimizer, 'optimizer_state_dict_best')
            exp.save_state(scheduler, 'scheduler_state_dict_best')
    return


def main(exp, config):
    logger = exp.logger
    model = config.model().to(device)
    loader = config.dataset().get_loader(train_shuffle=True)
    params = model.parameters()
    optimizer = config.optimizer(params)
    criterion = config.criterion()
    scheduler = config.scheduler(optimizer)
    start_epoch = 0

    if args.load_model:
        exp_stats = exp.load_epoch_stats()
        start_epoch = exp_stats['epoch'] + 1
        model = exp.load_state(model, 'model_state_dict')
        optimizer = exp.load_state(optimizer, 'optimizer_state_dict')
        scheduler = exp.load_state(scheduler, 'scheduler_state_dict')
    elif args.load_best_model:
        model = exp.load_state(model, 'model_state_dict_best')
        optimizer = exp.load_state(optimizer, 'optimizer_state_dict_best')
        scheduler = exp.load_state(scheduler, 'scheduler_state_dict_best')
    if args.data_parallel:
        model = torch.nn.DataParallel(model).to(device)
        logger.info("Using torch.nn.DataParallel")

    train(args, exp, model, optimizer, scheduler, criterion, loader,
          start_epoch)

    return


if __name__ == '__main__':
    args = parser.parse_args()
    args.eps = args.eps/255
    args.step_size = args.step_size/255
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # setup experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    config = mlconfig.load(config_filename)
    config.set_immutable()
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    logger = experiment.logger
    logger.info("PyTorch Version: %s" % (torch.__version__))

    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i)
                       for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    for key in config:
        logger.info("%s: %s" % (key, config[key]))

    start = time.time()
    main(experiment, config)
    end = time.time()

    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)
