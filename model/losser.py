import torch.nn as nn
import torch.nn.functional as F


class Losser(nn.Module):
    def __init__(self, cfg):
        super(Losser, self).__init__()
        self.cfg = cfg

    def forward(self, preds, targets):
        if self.cfg.TRAIN.LOSS_CRITERION == "L1":
            loss = F.l1_loss(preds, targets)
        elif self.cfg.TRAIN.LOSS_CRITERION == "MSE":
            loss = F.mse_loss(preds, targets)
        else:
            raise Exception(
                'Unknown criterion : {}'.format(
                    self.cfg.TRAIN.LOSS_CRITERION))

        return loss
