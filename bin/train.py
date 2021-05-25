import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from engine.trainer import Trainer

parser = argparse.ArgumentParser(description='Face rigger trainer')
parser.add_argument(
    'cfg_file',
    default=None,
    type=str,
    help='Model config in yaml format')
parser.add_argument(
    'save_path',
    default=None,
    type=str,
    help='Path to save model')
parser.add_argument(
    '--num_workers',
    default=1,
    type=int,
    help='Number of workers for each dataloader')
parser.add_argument(
    '--device_ids',
    default='0',
    type=str,
    help='GPU indices comma separated')
parser.add_argument(
    '--logtofile',
    default=False,
    type=bool,
    help='Save log to save_path/log.txt if set True')
parser.add_argument(
    '--local_rank',
    default=0,
    type=int,
    help='DDP parameter, do not modify')


def run(args):
    trainer = Trainer(
        args.cfg_file,
        args.save_path,
        num_workers=args.num_workers,
        device_ids=args.device_ids,
        logtofile=args.logtofile,
        local_rank=args.local_rank)
    cfg = trainer.cfg

    epoch_steps = len(trainer.dataloader_train)
    total_steps = epoch_steps * cfg.TRAIN.EPOCH
    start_step = trainer.summary['step']

    for step in range(start_step, total_steps):
        trainer.train_step()

        if (step + 1) % cfg.TRAIN.LOG_EVERY == 0:
            if args.local_rank == 0:
                trainer.logging('Train')
                trainer.write_summary('Train')
            trainer.log_init()

        if (step + 1) % cfg.TRAIN.DEV_EVERY == 0 and args.local_rank == 0:
            trainer.dev_epoch()
            trainer.logging('Dev')
            trainer.write_summary('Dev')
            trainer.save_model('Train')
            trainer.save_model('Dev')

    trainer.close()


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()