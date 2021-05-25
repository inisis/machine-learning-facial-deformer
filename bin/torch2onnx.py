import os
import torch
import argparse

from model.neural_facial_rigger import NeuralFacialRigger

parser = argparse.ArgumentParser(description='Face rigger model converter')
parser.add_argument(
    '--embedding_size',
    default=100,
    type=int,
    help='Embedding size')
parser.add_argument(
    '--num_vertices',
    type=int,
    default=95820,
    help='Number of vertices.')
parser.add_argument(
    '--pretrained_model',
    default='./train_hair/model/20210302-120027/120.pth',
    type=str,
    help='Pretrained model weights')
parser.add_argument(
    '--output_model_name',
    default='test_best_model.onnx',
    type=str,
    help='Output model name')
# 训练数据做PCA之后的均值文件
parser.add_argument(
    '--mean_mesh_file',
    type=str,
    default='D:/data/pca/mean.npy',
    help='the mean file of the training data')

# 训练数据做PCA之后的特征向量
parser.add_argument(
    '--feat_vec_file',
    type=str,
    default='D:/data/pca/featVec.npy',
    help='the feature vector of the training data')


def convert(args):
    dummy_input = torch.randn(1, 99, device='cpu')

    input_names = ["ctrl"]
    output_names = ["mesh"]

    net = NeuralFacialRigger(out_dim=args.embedding_size, num_vertices=args.num_vertices, mean_mesh_file=args.mean_mesh_file, feat_vec_file=args.feat_vec_file)
    net.load_state_dict(torch.load(args.pretrained_model))

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
