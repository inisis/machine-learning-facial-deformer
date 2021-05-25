import os
import time
import yaml
import sys
import logging
import numpy as np
from shutil import copyfile, copytree
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from data.dataset import CSVDataset
from model.neural_facial_rigger import NeuralFacialRigger
from model.utils import lr_schedule, get_optimizer
from model.losser import Losser

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class Trainer(object):
    def __init__(self, cfg_file, save_path, num_workers=1,
                 device_ids='0', logtofile=False, local_rank=-1):
        self.cfg_file = cfg_file
        self.save_path = save_path
        self.num_workers = num_workers
        self.device_ids = device_ids
        self.logtofile = logtofile
        self.local_rank = local_rank
        self._init_cfg()
        self._setup()
        self._init_menbers()

    def _init_cfg(self):
        with open(self.cfg_file, 'r', encoding='utf-8') as f:
            self.cfg = edict(yaml.full_load(f))

    def _lr_schedule(self):
        lr = lr_schedule(self.cfg.SOLVER.LR, self.cfg.SOLVER.LR_FACTOR, self.summary['epoch'], self.cfg.SOLVER.LR_EPOCHS)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _setup(self):
        if not os.path.exists(self.save_path) and self.local_rank == 0:
            os.mkdir(self.save_path)

        if self.logtofile is True and self.local_rank == 0:
            logging.basicConfig(
                filename=self.save_path +
                '/log.txt',
                filemode='a+',
                level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)

        if self.local_rank == 0:
            logging.info(
                '{} setup Trainer...'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S")))

        device_ids = list(map(int, self.device_ids.split(',')))
        num_devices = torch.cuda.device_count()
        if num_devices < len(device_ids):
            raise Exception(
                'available gpu : {} < --device_ids : {}'.format(num_devices, len(device_ids)))

        src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
        dst_folder = os.path.join(self.save_path, 'neuralfacialrig')

        copytree(src_folder, dst_folder)

        copyfile(self.cfg_file, os.path.join(self.save_path, 'cfg.yaml'))

        copyfile(
            self.cfg.TRAIN.TRAIN_CSV,
            os.path.join(
                self.save_path,
                'train.csv'))
        copyfile(
            self.cfg.TRAIN.DEV_CSV,
            os.path.join(
                self.save_path,
                'dev.csv'))

    def _init_menbers(self):
        cfg = self.cfg
        self.device_ids = list(map(int, self.device_ids.split(',')))
        self.device_id = self.device_ids[self.local_rank]
        self.device = torch.device('cuda:{}'.format(self.device_id))
        torch.cuda.set_device(self.device_id)

        if self.local_rank == 0:
            self.summary_writer = SummaryWriter(self.save_path)

        model = NeuralFacialRigger(cfg).to(self.device)

        if cfg.MODEL.PRETRAINED is not None:
            ckpt = torch.load(cfg.MODEL.PRETRAINED, map_location=torch.device('cpu'))
            model.load_state_dict(ckpt['state_dict'], strict=False)

        print(model)
        self.model = nn.DataParallel(model)

        self.losser = Losser(cfg).to(self.device).train()

        self.optimizer = get_optimizer(self.model.parameters(), cfg)

        self.dataset_train = CSVDataset(
            self.cfg.TRAIN.TRAIN_CSV, cfg, mode='train')

        if sys.platform is 'win32':
            self.dataloader_train = DataLoader(
                self.dataset_train,
                batch_size=self.cfg.TRAIN.TRAIN_BATCH_SIZE,
                drop_last=False,
                shuffle=True)
        else:
            self.dataloader_train = DataLoader(
                self.dataset_train,
                batch_size=self.cfg.TRAIN.TRAIN_BATCH_SIZE,
                num_workers=self.num_workers,
                drop_last=False,
                shuffle=True)

        self.dataiter_train = iter(self.dataloader_train)

        if sys.platform is 'win32':
            self.dataloader_dev = DataLoader(
                CSVDataset(
                    self.cfg.TRAIN.DEV_CSV,
                    cfg,
                    mode='dev'),
                batch_size=self.cfg.TRAIN.DEV_BATCH_SIZE,
                drop_last=False,
                shuffle=False)
        else:
            self.dataloader_dev = DataLoader(
                CSVDataset(
                    self.cfg.TRAIN.DEV_CSV,
                    cfg,
                    mode='dev'),
                batch_size=self.cfg.TRAIN.DEV_BATCH_SIZE,
                num_workers=self.num_workers,
                drop_last=False,
                shuffle=False)

        self.summary = {
            'step': 0,
            'log_step': 0,
            'epoch': 0,
            'loss_sum_train': 0,
            'loss_dev': float('inf'),
            'loss_dev_best': float('inf')}

        self.time_now = time.time()
        logging.info(
            '{} start training...'.format(
                time.strftime("%Y-%m-%d %H:%M:%S")))

    def log_init(self):
        self.summary['loss_sum_train'] = 0
        self.summary['log_step'] = 0

    def train_step(self):
        try:
            ctrl, mesh_vertex = next(self.dataiter_train)
        except StopIteration:
            self.summary['epoch'] += 1
            self._lr_schedule()
            self.dataiter_train = iter(self.dataloader_train)
            ctrl, mesh_vertex = next(self.dataiter_train)

        ctrl = ctrl.to(self.device)
        mesh_vertex = mesh_vertex.to(self.device)

        pred = self.model(ctrl)

        loss = self.losser(pred, mesh_vertex)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary['step'] += 1
        self.summary['loss_sum_train'] = loss
        self.summary['log_step'] += 1

    def dev_epoch(self):
        torch.set_grad_enabled(False)
        self.time_now = time.time()
        self.model.eval()
        self.losser.eval()

        steps = len(self.dataloader_dev)
        dataiter = iter(self.dataloader_dev)

        loss_sum = 0
        for step in range(steps):
            ctrl, mesh_vertex = next(dataiter)

            ctrl = ctrl.to(self.device)
            mesh_vertex = mesh_vertex.to(self.device)

            pred = self.model(ctrl)
            loss = self.losser(pred, mesh_vertex)

            loss_sum += loss

        self.summary['loss_dev'] = loss_sum / steps

        torch.set_grad_enabled(True)
        self.model.train()
        self.losser.train()

    def logging(self, mode='Train'):
        time_spent = time.time() - self.time_now
        self.time_now = time.time()

        if mode == 'Train':
            loss_train = self.summary['loss_sum_train'] / \
                self.summary['log_step']

            loss_train = loss_train.detach().cpu().numpy()
            logging.info(
                '{}, Train, Epoch {}, Step : {}, Loss : {}, Run Time : {:.2f} sec'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.summary['epoch'],
                    self.summary['step'],
                    str(loss_train),
                    time_spent))
        elif mode == 'Dev':
            loss_dev = self.summary['loss_dev'].detach().cpu().numpy()

            logging.info(
                '{}, Dev, Epoch {}, Step : {}, Loss : {}, Run Time : {:.2f} sec'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.summary['epoch'],
                    self.summary['step'],
                    str(loss_dev),
                    time_spent))

    def write_summary(self, mode='Train'):
        if mode == 'Train':
            self.summary_writer.add_scalar(
                'Train/loss',
                (self.summary['loss_sum_train'] /
                self.summary['log_step']).detach().cpu().numpy(),
                self.summary['step'])
        elif mode == 'Dev':
            self.summary_writer.add_scalar(
                'Dev/loss',
                self.summary['loss_dev'].detach().cpu().numpy(),
                self.summary['step'])

    def save_model(self, mode='Train'):
        save_best = False

        if mode == 'Train':
            torch.save(
                {
                    'epoch': self.summary['epoch'],
                    'step': self.summary['step'],
                    'loss_dev_best': self.summary['loss_dev_best'],
                    'state_dict': self.model.module.state_dict()},
                os.path.join(
                    self.save_path,
                    'train.ckpt'))
        elif mode == 'Dev':
            save_best = False
            if self.summary['loss_dev'] < self.summary['loss_dev_best']:
                self.summary['loss_dev_best'] = self.summary['loss_dev']
                save_best = True

        if save_best:
            torch.save(
                {
                    'epoch': self.summary['epoch'],
                    'step': self.summary['step'],
                    'loss_dev_best': self.summary['loss_dev_best'],
                    'state_dict': self.model.module.state_dict()},
                os.path.join(
                    self.save_path,
                    'best.ckpt'))

            loss_dev = self.summary['loss_dev'].detach().cpu().numpy()

            logging.info(
                '{}, Best, Epoch {}, Step : {}, Loss : {}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.summary['epoch'],
                    self.summary['step'],
                    str(loss_dev)))

    def close(self):
        if self.local_rank == 0:
            self.summary_writer.close()