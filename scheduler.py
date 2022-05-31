import numpy as np


class Scheduler:
    def get_lr(self, step_idx: int, cycle_len: int):
        raise "Not implemented yet"


class FGEScheduler(Scheduler):
    def __init__(self, lr_1: float, lr_2: float):
        self.lr_1 = lr_1
        self.lr_2 = lr_2

    def get_lr(self, step_idx: int, cycle_len: int):
        t_i = ((step_idx - 1) % cycle_len + 1.0) / cycle_len

        if t_i <= 0.5:
            return (1 - 2 * t_i) * self.lr_1 + 2 * t_i * self.lr_2
        else:
            return (2 - 2 * t_i) * self.lr_2 + (2 * t_i - 1) * self.lr_1


class SSEScheduler(Scheduler):
    def __init__(self, alpha_0: float):
        self.alpha_0 = alpha_0

    def get_lr(self, step_idx: int, cycle_len: int):
        t_i = float((step_idx - 1) % cycle_len) / cycle_len
        return self.alpha_0 * (np.cos(np.pi * t_i) + 1) / 2
