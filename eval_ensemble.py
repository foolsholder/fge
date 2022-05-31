from argparse import ArgumentParser

from ensemble_runner import EnsemblePredictor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_cfg = {
        'num_classes': args.num_classes,
        'depth': 164,
        'device': 'cuda:0'
    }
    folders = [
        '../ckpt_ind_1',
        '../ckpt_ind_2',
        '../ckpt_ind_3',
    ]
    predictor = EnsemblePredictor(model_cfg, folders=folders)
    print('ensemble accuracy: {:.4f}'.format(predictor.evaluate()))
