import os
import torch
import wandb


from torch.utils.data import DataLoader
from torch.nn import functional as F
from typing import Union, List, Dict
from tqdm.auto import trange


from data import get_loader
from preresnet import create_model
from runner import SingleRunner
from scheduler import Scheduler
from tqdm.auto import tqdm


class EnsembleTrainer(SingleRunner):
    def __init__(self,
                 model_cfg: Dict[str, Union[int, str]],
                 lr_scheduler: Scheduler):
        super().__init__(model_cfg)
        self.lr_scheduler = lr_scheduler
        self.cycle_len = 0

    def manage_lr(self):
        lr = self.lr_scheduler.get_lr(self.step, self.cycle_len)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.log_metric('train_details', 'lr', lr)

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.manage_lr()
        self.optimizer.step()

    def restore_parameters(self):
        path = os.path.join(self.model_cfg['ckpt_restore'], '{}')
        self.model.load_state_dict(torch.load(path.format('model_ckpt_150.pth')))
        self.optimizer.load_state_dict(torch.load(path.format('opt_ckpt_150.pth')))

    def train(
            self,
            train_cfg: Dict[str, int],
            project_name: str = 'fge',
            experiment_name: str = 'ensemble'
        ) -> None:
        self.model.train()
        self.set_optimizer()
        self.restore_parameters()

        self.set_data_generator()
        epoch_len = len(self.train_loader.dataset) // self.train_loader.batch_size
        self.cycle_len = train_cfg['cycle_len_epoch'] * epoch_len
        shift_epoch = train_cfg['shift_epoch']

        self.step = 0

        wandb.init(project=project_name, name=experiment_name)
        for epoch in trange(1, 1 + train_cfg['epochs']):
            self.epoch = epoch
            self.run_loader(self.train_loader, 'train')
            with torch.no_grad():
                self.run_loader(self.valid_loader, 'valid')
            self.log_metric('train_details', 'epoch', epoch)

            if (epoch - shift_epoch) % train_cfg['cycle_len_epoch'] == 0:
                self.save_checkpoint(self.model_cfg['ckpt_folder'])


        self.model.eval()


class EnsemblePredictor:
    def __init__(self, model_cfg: Dict[str, Union[int, str]], folders: List[str]):
        self.models: List[torch.nn.Module] = []
        self.device = model_cfg['device']

        from glob import glob
        for folder in folders:
            for model_ckpt in glob(os.path.join(folder, 'model_ckpt*.pth')):
                model = create_model(model_cfg)
                model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
                model.to(self.device)
                model.eval()
                self.models += [model]

        self.valid_loader = get_loader(
            num_classes=model_cfg['num_classes'],
            train=False,
            batch_size=128,
            shuffle=False,
            drop_last=False
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        total_acc = 0.
        total_count = 0
        for (X, y) in tqdm(self.valid_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            batch_size = X.shape[0]

            pred_sum_proba = None
            for model in self.models:
                logits = model(X)
                proba = F.softmax(logits, dim=1)
                if pred_sum_proba is None:
                    pred_sum_proba = proba
                else:
                    pred_sum_proba += proba
            pred_labels = torch.argmax(pred_sum_proba, dim=1)
            total_acc += torch.sum((pred_labels == y).float())
            total_count += batch_size

        return total_acc / total_count
