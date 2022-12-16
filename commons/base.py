import torch
from torch.nn import Module


class Exemplar(Module):
    def __init__(self, space, idx, genotype, gen=0):
        super().__init__()

        self.space = space
        self.idx = idx
        self.generation = gen
        self.genotype = genotype

        self.rank = None

        self.born = False

        self._cost_info = None
        self._metrics = None
        self.val_accuracy = None

    def get_metric(self, metric_name: str) -> float:
        return self.space._get_metric_val(self.idx, metric_name)

    def get_cost_info(self):
        if self._cost_info is not None:
            return self._cost_info
        else:
            self._cost_info = self.space.get_cost_info(self.idx)
        return self._cost_info

    def skip(self) -> int:
        return self.space._skip(self)

    def set_generation(self, gen):
        self.generation = gen

        return self

    def get_accuracy(self):
        return self.space.get_accuracy(self)

    def get_val_accuracy(self):
        if self.val_accuracy is None:
            self.val_accuracy = self.space.get_val_accuracy(self)
        return self.val_accuracy

    def get_network(self, device: torch.device) -> Module:
        return self.space.get_network(self, device=device)


METRIC_NAMES = {'naswot'}
