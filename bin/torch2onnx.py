import os
import sys
import yaml
import torch
import argparse
from easydict import EasyDict as edict

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from model.neural_facial_rigger import NeuralFacialRigger

parser = argparse.ArgumentParser(description='Face rigger model converter')
parser.add_argument(
    'pretrained_model',
    default=None,
    type=str,
    help='Pretrained model weights')
parser.add_argument(
    'cfg_file',
    default=None,
    type=str,
    help='Model config in yaml format')
parser.add_argument(
    '--output_model_name',
    default='test_best_model.onnx',
    type=str,
    help='Output model name')


def convert(args):

    with open(args.cfg_file) as f:
        cfg = edict(yaml.full_load(f))

    dummy_input = torch.randn(1, cfg.TRAIN.NUM_RIG_PARAM, device='cpu')

    input_names = ["ctrl"]
    output_names = ["mesh"]

    net = NeuralFacialRigger(cfg)
    net.load_state_dict(torch.load(args.pretrained_model)['state_dict'])

    save_path = os.path.dirname(args.pretrained_model)

    torch.onnx.export(
        net,
        dummy_input,
        os.path.join(save_path, args.output_model_name),
        verbose=True,
        input_names=input_names,
        output_names=output_names)


if __name__ == '__main__':
    args = parser.parse_args()
    convert(args)
