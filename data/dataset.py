import numpy as np

from torch.utils.data import Dataset

np.random.seed(0)


class CSVDataset(Dataset):
    def __init__(self, label_file, cfg, mode='train'):
        self.cfg = cfg
        self._mode = mode
        self.ctrl_files = []
        self.mesh_vertex_files = []

        with open(label_file) as f:
            for line in f:
                fields = line.strip('\n').split(',')
                ziva_file = fields[1]
                mesh_vertex_file = fields[0]
                self.ctrl_files.append(ziva_file)
                self.mesh_vertex_files.append(mesh_vertex_file)

        self._num_samples = len(self.ctrl_files)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        ziva = np.load(self.ctrl_files[idx]).astype(np.float32)
        mesh_vertex = np.load(self.mesh_vertex_files[idx]).astype(np.float32)
        path = self.ctrl_files[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (ziva, mesh_vertex)
        elif self._mode == 'test':
            return (ziva, path)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
