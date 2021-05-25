import os
import time
import argparse
import numpy as np

from sklearn.decomposition import PCA


def get_parser():
    parser = argparse.ArgumentParser(description='Principal Component Analysis')

    # 设置训练数据集路径
    parser.add_argument('--train_file', type=str,
                        default='D:/data/csv/pca.csv',
                        help='The file path of the train data file')

    # 设置模型的训练轮数
    parser.add_argument('--n_components', type=int,
                        default=300,
                        help='The max epoch to train the network')

    args = parser.parse_args()
    return args


def run(train_files, n_components, save_path=None):
    start_time = time.time()
    if not isinstance(train_files, (list, tuple)):
        train_files = (train_files,)
    mesh_list = []
    if save_path is None:
        save_path = os.path.dirname(train_files[0])
    for train_file in train_files:
        end_time = time.time()
        print("train_file processing time:%d" % (end_time - start_time))
        with open(train_file) as f:
            for line in f:
                fields = line.strip('\n').split(',')
                mesh_vertex_file = fields[0]
                mesh_vertex = np.load(mesh_vertex_file)
                mesh_list.append(mesh_vertex)

    data = np.array(mesh_list)

    mean = np.mean(data, axis=0)
    np.save(os.path.join(save_path, "mean.npy"), mean)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    end_time = time.time()
    print("Time:%d" % (end_time - start_time))
    np.save(os.path.join(save_path, "featVec.npy"), pca.components_)
    print("Done")


def main():
    args = get_parser()
    run(args)


if __name__ == '__main__':
    main()
