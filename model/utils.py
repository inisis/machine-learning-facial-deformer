import numpy as np
from torch.optim import SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop


def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):

    count = sum(epoch_now >= epoch for epoch in lr_epochs)

    return lr * np.power(lr_factor, count)


def get_optimizer(params, cfg):
    if cfg.SOLVER.OPTIMIZER == 'SGD':
        return SGD(params, lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM,
                   weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'Adadelta':
        return Adadelta(params, lr=cfg.SOLVER.LR,
                        weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'Adagrad':
        return Adagrad(params, lr=cfg.SOLVER.LR,
                       weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'Adam':
        return Adam(params, lr=cfg.SOLVER.LR,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'AdamW':
        return AdamW(params, lr=cfg.SOLVER.LR,
                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'RMSprop':
        return RMSprop(params, lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM,
                       weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise Exception('Unknown optimizer: {}'.format(cfg.SOLVER.OPTIMIZER))