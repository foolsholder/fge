import os

from argparse import ArgumentParser

from ensemble_runner import EnsembleTrainer
from scheduler import FGEScheduler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--ckpt_restore', type=str, default='ckpt_ind_1')
    parser.add_argument('--ckpt_folder', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='fge')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_cfg = {
        'num_classes': args.num_classes,
        'depth': 164,
        'ckpt_restore': args.ckpt_restore,
        'ckpt_folder': args.ckpt_folder,
        'device': 'cuda:0'
    }
    train_cfg = {
        'epochs': 22,
        'cycle_len_epoch': 4,
        'shift_epoch': 2
    }

    if not os.path.exists(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    runner = EnsembleTrainer(model_cfg, FGEScheduler(lr_1=5e-2, lr_2=5e-4))
    runner.train(train_cfg, experiment_name=args.exp_name)
