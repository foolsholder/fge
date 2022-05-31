import os

from argparse import ArgumentParser

from runner import RestoreRunner


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--ckpt_folder', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='ind')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_cfg = {
        'num_classes': args.num_classes,
        'depth': 164,
        'ckpt_folder': args.ckpt_folder,
        'device': 'cuda:0'
    }
    train_cfg = {
        'epochs': 150
    }

    if not os.path.exists(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    runner = RestoreRunner(model_cfg)
    runner.train(train_cfg, experiment_name=args.exp_name)
