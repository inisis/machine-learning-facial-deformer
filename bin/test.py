import os
import sys
import yaml
import argparse
import logging
from statistics import mean
from easydict import EasyDict as edict

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

import torch
from torch.utils.data import DataLoader

from data.dataset import CSVDataset
from model.losser import Losser
from model.neural_facial_rigger import NeuralFacialRigger

parser = argparse.ArgumentParser(description='Face rigger test')
parser.add_argument(
    'ckpt_file',
    default=None,
    type=str,
    help='CKPT file to the trained model')
parser.add_argument(
    'cfg_file',
    default=None,
    type=str,
    help='Model config in yaml format')
parser.add_argument(
    'csv_file',
    default=None,
    type=str,
    help='CSV file to the input data')
parser.add_argument(
    '--neutral_head_mesh',
    type=str,
    default="D:/data/ray/neutralHead.npy",
    help='Path for the neutralHead numpy data.')
parser.add_argument(
    '--num_workers',
    default=1,
    type=int,
    help='Number of workers for each dataloader')


def calculate_dist(pred, gt):
    if pred.shape != gt.shape:
        print("The shape predict and truth must same")

    predict = pred.reshape(-1, 3)
    truth = gt.reshape(-1, 3)

    dist = np.linalg.norm(predict - truth, axis=1)
    max_dist = max(dist)
    mean_dist = mean(dist)

    return max_dist, mean_dist


def test_epoch(args, cfg, device, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    steps = len(dataloader)
    dataiter = iter(dataloader)

    error_list = []
    max_dist_list = []
    mean_dist_list = []

    diffs = []
    preds = []
    gts = []

    neutral_head_mesh = np.load(args.neutral_head_mesh).astype(np.float32)

    losser = Losser(cfg).to(device)

    for step in range(steps):
        ziva, mesh_vertex = next(dataiter)
        ziva = ziva.to(device)
        pred = model(ziva)

        loss = losser(pred, mesh_vertex)

        loss = loss.detach().cpu().numpy()
        error_list.append(loss)

        pred_array = pred.detach().cpu().numpy().astype("float64")
        mesh_array = mesh_vertex.cpu().data.numpy().astype("float64")

        max_dist, mean_dist = calculate_dist(pred_array, mesh_array)

        diffs.append(pred - mesh_vertex)
        preds.append(pred + neutral_head_mesh)
        gts.append(mesh_vertex + neutral_head_mesh)

        max_dist_list.append(max_dist)
        mean_dist_list.append(mean_dist)

    logging.info("MSE error mean: " +
                 str(sum(error_list)/len(error_list)) +
                 " Mean point distance: " +
                 str(sum(mean_dist_list)/len(mean_dist_list)))
    logging.info("MSE error max: " + str(max(error_list)) +
                 " Max point distance: " + str(max(max_dist_list)))

    save_name = args.ckpt_file.split("/")[-2]

    diffs = np.concatenate(diffs)
    np.save('./' + save_name + '_difference.npy', diffs)

    preds = np.concatenate(preds)
    np.save('./' + save_name + '_pred.npy', preds)

    gts = np.concatenate(gts)
    np.save('./' + save_name + '_points.npy', gts)


def run(args):
    with open(args.cfg_file) as f:
        cfg = edict(yaml.full_load(f))

    device = torch.device('cpu')
    model = NeuralFacialRigger(cfg).to(device)
    ckpt = torch.load(args.ckpt_file, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    data_loader_test = DataLoader(CSVDataset(cfg.TRAIN.DEV_CSV, cfg, mode='dev'),
                                  batch_size=1,
                                  num_workers=args.num_workers,
                                  drop_last=False,
                                  shuffle=False)

    test_epoch(args, cfg, device, model, data_loader_test)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
