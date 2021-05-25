from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cfg, in_dim = 256):
        super(BasicBlock, self).__init__()
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.fc = nn.Linear(in_features=in_dim, out_features=256)

    def forward(self, x):
        identity = x
        out = self.leaky_relu(x)
        out = self.fc(out)
        out = out + identity

        return out


class NeuralFacialRigger(nn.Module):

    def __init__(self, cfg):
        super(NeuralFacialRigger, self).__init__()
        self.cfg = cfg
        self.layer1 = self._make_layer(self.cfg.TRAIN.NUM_RIG_PARAM, self.cfg.TRAIN.BASE_DEPTH, BasicBlock)
        self.layer2 = nn.Linear(in_features=256,
                                out_features=self.cfg.TRAIN.EMBEDDING_SIZE)
        self.layer3 = nn.Linear(
            in_features=self.cfg.TRAIN.EMBEDDING_SIZE,
            out_features=self.cfg.TRAIN.MESH_VERTEX_SIZE)

    def _make_layer(self, num_rig_param, depth, block):
        layers = []
        layers.append(nn.Linear(in_features=num_rig_param, out_features=256))
        for _ in range(1, depth):
            layers.append(block(self.cfg))

        return nn.Sequential(*layers)

    def forward(self, ziva):
        x = self.layer1(ziva)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
