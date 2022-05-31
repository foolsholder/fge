import os
import torch
import wandb


from torch.utils.data import DataLoader
from torch.nn import functional as F
from typing import Union, List, Dict
from tqdm.auto import trange


from data import get_loader
from preresnet import create_model


class SingleRunner:
    def __init__(
            self,
            model_cfg: Dict[str, Union[int, str]]
    ):
        self.model_cfg = model_cfg
        self.model = create_model(model_cfg)

        device = torch.device(self.model_cfg['device'])
        self.device = device
        self.model.to(device)

    def set_optimizer(self) -> None:
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.optimizer = optimizer

    def calc_loss(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(X)
        ce_loss = F.cross_entropy(logits, y)
        pred_labels = torch.argmax(logits, dim=1)
        accuracy = torch.mean((pred_labels == y).float())
        return {
            'loss': ce_loss,
            'accuracy':  accuracy
        }

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_loader(self, loader: DataLoader, loader_name: str):
        self.model.train(loader_name == 'train')
        total_loss = 0.
        total_acc = 0.
        total_count = 0

        for (X, y) in loader:
            X = X.to(self.device)
            y = y.to(self.device)
            batch_size = X.shape[0]

            scores = self.calc_loss(X, y)
            loss = scores.pop('loss')
            accuracy = scores.pop('accuracy')

            total_loss += loss.item() * batch_size
            total_acc += accuracy.item() * batch_size
            total_count += batch_size

            if loader_name == 'train':
                self.step += 1
                self.optimizer_step(loss)
                self.log_metric('loss', 'train', loss)
                self.log_metric('accuracy', 'train', accuracy)

        self.log_metric('loss_epoch', loader_name, total_loss / total_count)
        self.log_metric('accuracy_epoch', loader_name, total_acc / total_count)

    def train(
            self,
            train_cfg: Dict[str, int],
            project_name: str = 'fge',
            experiment_name: str = 'ind'
        ) -> None:
        self.set_optimizer()
        self.set_data_generator()
        self.step = 0

        wandb.init(project=project_name, name=experiment_name)
        self.model.train()

        for epoch in trange(1, 1 + train_cfg['epochs']):
            self.epoch = epoch
            self.run_loader(self.train_loader, 'train')
            with torch.no_grad():
                self.run_loader(self.valid_loader, 'valid')
            self.log_metric('train_details', 'epoch', epoch)

        self.model.eval()
        self.save_checkpoint(self.model_cfg['ckpt_folder'])

    def save_checkpoint(self, ckpt_folder: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(ckpt_folder, 'model_ckpt_{}.pth'.format(self.epoch)))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_folder, 'opt_ckpt_{}.pth'.format(self.epoch)))

    def set_data_generator(self):
        self.train_loader = get_loader(
            num_classes=self.model_cfg['num_classes'],
            train=True,
            batch_size=128,
            shuffle=True,
            drop_last=True
        )
        self.valid_loader = get_loader(
            num_classes=self.model_cfg['num_classes'],
            train=False,
            batch_size=128,
            shuffle=False,
            drop_last=False
        )


class RestoreRunner(SingleRunner):
    def restore_parameters(self):
        path = os.path.join(self.model_cfg['ckpt_folder'], '{}')
        self.model.load_state_dict(torch.load(path.format('model_ckpt_150.pth')))
        self.optimizer.load_state_dict(torch.load(path.format('opt_ckpt_150.pth')))

        for group in self.optimizer.param_groups:
            group['lr'] = 1e-2

    def train(
            self,
            train_cfg: Dict[str, int],
            project_name: str = 'fge',
            experiment_name: str = 'ind'
        ) -> None:
        self.set_optimizer()
        self.restore_parameters()
        self.set_data_generator()
        self.step = 0

        wandb.init(project=project_name, name=experiment_name)
        self.model.train()

        for epoch in trange(1 + train_cfg['epochs'], 1 + train_cfg['epochs'] + 25):
            self.epoch = epoch
            self.run_loader(self.train_loader, 'train')
            with torch.no_grad():
                self.run_loader(self.valid_loader, 'valid')
            self.log_metric('train_details', 'epoch', epoch)

        self.model.eval()
        self.save_checkpoint(self.model_cfg['ckpt_folder'])
