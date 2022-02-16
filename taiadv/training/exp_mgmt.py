# Copyright (c) OpenTAI. All rights reserved.
import datetime
import json
import os
import shutil

import torch

from . import util


class ExperimentManager():

    def __init__(self, exp_name, exp_path, config_file_path=None):
        if exp_name == '' or exp_name is None:
            exp_name = 'exp_at' + datetime.datetime.now()
        exp_path = os.path.join(exp_path, exp_name)
        checkpoint_path = os.path.join(exp_path, 'checkpoints')
        log_filepath = os.path.join(exp_path, exp_name) + '.log'
        stas_hist_path = os.path.join(exp_path, 'stats')
        stas_eval_path = os.path.join(exp_path, 'stats_eval')

        util.build_dirs(exp_path)
        util.build_dirs(checkpoint_path)
        util.build_dirs(stas_hist_path)
        util.build_dirs(stas_eval_path)

        if config_file_path is not None:
            shutil.copyfile(config_file_path,
                            os.path.join(exp_path, exp_name + '.yaml'))

        self.exp_name = exp_name
        self.exp_path = exp_path
        self.checkpoint_path = checkpoint_path
        self.log_filepath = log_filepath
        self.stas_hist_path = stas_hist_path
        self.stas_eval_path = stas_eval_path
        self.logger = util.setup_logger(name=exp_path, log_file=log_filepath)

    def save_eval_stats(self, exp_stats, name):
        filename = '%s_exp_stats_eval.json' % name
        filename = os.path.join(self.stas_eval_path, filename)
        with open(filename, 'w') as outfile:
            json.dump(exp_stats, outfile)
        return

    def load_eval_stats(self, name):
        filename = '%s_exp_stats_eval.json' % name
        filename = os.path.join(self.stas_eval_path, filename)
        if os.path.exists(filename):
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                return data
        else:
            return None

    def save_epoch_stats(self, epoch, exp_stats):
        filename = 'exp_stats_epoch_%d.json' % epoch
        filename = os.path.join(self.stas_hist_path, filename)
        with open(filename, 'w') as outfile:
            json.dump(exp_stats, outfile)
        return

    def load_epoch_stats(self, epoch=None):
        if epoch is not None:
            filename = 'exp_stats_epoch_%d.json' % epoch
            filename = os.path.join(self.stas_hist_path, filename)
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                return data
        else:
            epoch = self.config.epochs
            filename = 'exp_stats_epoch_%d.json' % epoch
            filename = os.path.join(self.stas_hist_path, filename)
            while not os.path.exists(filename) and epoch >= 0:
                epoch -= 1
                filename = 'exp_stats_epoch_%d.json' % epoch
                filename = os.path.join(self.stas_hist_path, filename)

            if not os.path.exists(filename):
                return None

            with open(filename, 'rb') as json_file:
                data = json.load(json_file)
                return data
        return None

    def save_state(self, target, name):
        if isinstance(target, torch.nn.DataParallel):
            target = target.module
        filename = os.path.join(self.checkpoint_path, name) + '.pt'
        torch.save(target.state_dict(), filename)
        self.logger.info('%s saved at %s' % (name, filename))
        return

    def load_state(self, target, name):
        filename = os.path.join(self.checkpoint_path, name) + '.pt'
        target.load_state_dict(torch.load(filename), strict=False)
        self.logger.info('%s loaded from %s' % (name, filename))
        return target
