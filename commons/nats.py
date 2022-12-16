from copy import deepcopy as copy
import os.path
from typing import Union, Text, Sequence
import numpy as np
import torch
from torch.nn import Module
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
from . import Exemplar
from . import load_metrics


class _NetWrapper(Module):

    def __init__(self, net: Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, get_ints: bool = False):
        # applying the whole network forward model
        x = self.net(x)
        # extracting the actual output from forward of given object
        if get_ints:
            return x[1], x[0]
        else:
            return x[1]


class NATSBench:
    _all_ops = {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3'}

    def __init__(self, path: str, dataset: str, verbose: bool = False, metric_root=None):
        # Value check the dataset
        known_datasets = ('cifar10', 'cifar100', 'imagenet16-120')
        _dataset = dataset.lower()
        if _dataset not in known_datasets:
            raise ValueError(f"Unknown dataset {dataset}. Dataset must be one of: {', '.join(known_datasets)}.")
        self._dataset = _dataset

        # Init wrapped API
        self._official_api = create(path, 'tss', fast_mode=True, verbose=verbose)

        # Load metrics
        if metric_root is not None:
            self.metrics_cache = load_metrics(os.path.join(metric_root, dataset))

    @staticmethod
    def get_api_name():
        return 'nats'

    def __len__(self):
        return len(self._official_api)

    def __iter__(self):
        # Returns an iterator
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item: int) -> Exemplar:
        # In the original wrapped API __getitem__ returns just the architecture string, without configuration
        # depending on the dataset

        config = self._official_api.get_net_config(item, self._dataset)
        return Exemplar(space=self, idx=item, genotype=config['arch_str'])

    def get_index(self, architecture: Union[Exemplar, Text]) -> int:
        """
        This function returns the index of input architecture 'architecture' in NATS Bench.

        :param architecture: Either an Exemplar object corresponding to the input architecture or a string
                  formatted according to NATS api.
        :return: Index at which it is possible to find `architecture` in the space considered.
        """
        if isinstance(architecture, Exemplar):
            genotype = architecture.genotype  # this gets the architecture
        elif isinstance(architecture, str):
            genotype = architecture
        else:
            raise ValueError(f"Invalid argument {architecture}")

        index = self._official_api.query_index_by_arch(genotype)
        if index < 0:
            raise KeyError(f"Unknown architeture: {genotype}")

        return index

    def get_accuracy(self, architecture: Union[Exemplar, Text], epochs: int = 200) -> float:
        """
        This function returns the accuracy value of a given architecture `architecture` when trained for
        `epochs` training epochs.
        :param architecture: Either an Exemplar object corresponding to the input architecture or a string
                  formatted according to NATS api.
        :param epochs: Number of training epochs.
        :return: Accuracy value.
        """
        index = self.get_index(architecture=architecture)
        info = self._official_api.get_more_info(index, self._dataset, hp=epochs, is_random=False)
        return info['test-accuracy']

    def get_val_accuracy(self, architecture: Union[Exemplar, Text], epochs: int = 12) -> float:  # or 90 epochs
        """
        This function returns the validation accuracy value of a given architecture `architecture` when trained for
        `epochs` training epochs.
        :param architecture: Either an Exemplar object corresponding to the input architecture or a string
                  formatted according to NATS api.
        :param epochs: Number of training epochs
        :return: Validation Accuracy value.
        """
        index = self.get_index(architecture=architecture)
        # simulate_train_eval returns
        # [validation_accuracy, latency, time_cost, current_total_time_cost]
        validation_accuracy, _, _, _ = \
            self._official_api.simulate_train_eval(index, dataset=self._dataset, hp=epochs)
        return validation_accuracy

    def get_cost_info(self, architecture: Union[Exemplar, int]) -> dict:
        """
        This function returns the hardware metrics for a given architecture `architecture`.
        :param architecture: Either an Exemplar object corresponding to the input architecture or an integer
                             indexing the architecture in the search space considered.
        :return: Hardware metrics dictionary.
        """
        # restricting the analysis to number of parameters, latency of the architecture and flops
        hardware_metrics = ["params", "latency", "flops"]

        if isinstance(architecture, Exemplar):
            index = self.get_index(architecture=architecture)
        elif isinstance(architecture, int):
            index = architecture
        else:
            raise ValueError(f"Invalid argument {architecture}. Type: {type(architecture)}")
        # retrieving the full cost as per NATS bench API
        full_cost_info = self._official_api.get_cost_info(index, self._dataset)
        # hardware metrics dictionary obtained selecting specific metrics from performance
        # on hardware metrics
        metrics = {
            metric: full_cost_info[metric] for metric in hardware_metrics
        }

        return metrics

    def total_metrics_time(self, exemplars: Sequence[Exemplar], metrics: Sequence[str]) -> float:
        """
        This function returns the sum of all the metrics for all the exemplars considered.
        :param exemplars: Sequence of various exemplars considered
        :param metrics: Metrics of interest
        :return: Sum of all metrics for all architectures
        """
        # indexes related to the architectures that have to be evaluated
        indexes = [self.get_index(architecture=exemplar) for exemplar in exemplars]
        # metrics considered in this experiment
        metrics = [metric + '_time=[s]' for metric in metrics if metric != 'skip' and metric != 'params']

        return self.metrics_cache.loc[indexes, metrics].sum(axis=1).sum(axis=0)

    def total_train_and_eval_time(self, exemplars: Sequence[Exemplar], epochs: int = 12) -> float:
        """
        This function returns the total time needed to train a group of exemplars for `epochs` and then test this
        sequence on validation set.
        :param exemplars: Sequence of various exemplars considered
        :param epochs: Number of training epochs
        :return: total time spent
        """
        indexes = [self.get_index(architecture=exemplar)for exemplar in exemplars]
        time = 0.0
        for index in indexes:
            _, _, time_cost, _ = self._official_api.simulate_train_eval(index, dataset=self._dataset, hp=epochs)
            time += time_cost
        return time

    def get_network(self, architecture: Exemplar, device: torch.device = None) -> Module:
        """
        This function interfaces objects of class Exemplar with PyTorch objects.
        :param architecture: Object of Exemplar class
        :param device: Device on which to run the model (CPU/GPU)
        :return: torch.nn.Module object
        """
        # using function of xautodl to obtain the model
        net = get_cell_based_tiny_net(self._official_api.get_net_config(architecture.idx, self._dataset))

        net = _NetWrapper(net)

        if device is not None:
            net = net.to(device)

        return net

    def _get_metric_val(self, index: int, metric_name: str):
        return self.metrics_cache.at[index, metric_name]

    @staticmethod
    def _skip(exemplar: Exemplar) -> float:
        """
        This function returns the fraction of skipped nodes over the total number of nodes in each cell.
        :param exemplar: Architecture considered
        :return: fraction of skipped nodes over total number of skip connections
        """
        levels = exemplar.genotype.split('+')
        max_len = 0
        counter = 0
        # looping over levels in the architecture
        for idx, level in enumerate(levels):
            level = level.split('|')[1:-1]
            n_genes = len(level)
            # looping over number of nodes involved in each level
            for i in range(n_genes):
                if 'skip' in level[i]:
                    counter += 1  # counting number of skip connections
                    min_edge = idx - i  # counting skipped nodes measuring the distance between current node and idx
                    max_len += min_edge  # summing up to obtain final number of
        if counter:
            return max_len / counter
        return 0.

    @staticmethod
    def _get_different_gene(gene: str):
        """
        This function returns a gene in the genotype different from gene.
        :param gene: String encoding the single gene considered
        :return: Chosing random gene from the ones still available
        """
        suitable = copy(NATSBench._all_ops)
        suitable.remove(gene)
        # choosing a candidate gene at random (uniformly)
        gene = np.random.choice(list(suitable))
        return gene + '~'

    def mutation(self, exemplar: Exemplar, R: int = 1) -> Exemplar:
        """
        This function mutates an exemplar in `R` different loci.
        :param exemplar: Actual architecture to mutate in this specific setting
        :param R: Number of different loci to modify
        :return: Mutated exemplar
        """
        # this indentifies a single individual
        genotype = exemplar.genotype.split('+')
        # collecting all the possible elements in the genotype
        levels = []
        for index, level in enumerate(genotype):
            level = level.split('|')[1:-1]
            levels.append(level)

        # modify R genes
        for _ in range(R):
            # the choice of the level to mutate is weighted by the number of genes per level
            chosen_level = np.argmax(
                np.random.multinomial(1, [1/6, 2/6, 3/6])  # extract one sample of size 3 from multinomial
            )
            # in each level, there are chosen_level + 1 different genes (e.g., level 0: 1 gene)
            mutating_gene = np.random.randint(chosen_level + 1)
            # replacing the considered level
            levels[chosen_level][mutating_gene] = self._get_different_gene(
                levels[chosen_level][mutating_gene][:-2]) + str(
                mutating_gene)

        for idx, level in enumerate(levels):
            levels[idx] = '|' + '|'.join(level) + '|'
        arch_string = '+'.join(levels)

        index = self.get_index(arch_string)
        return self[index]

    def crossover(self, exemplars: Sequence[Exemplar]):
        assert len(exemplars) == 2

        genotype1 = exemplars[0].genotype
        genotype2 = exemplars[1].genotype

        levels1 = genotype1.split('+')
        levels2 = genotype2.split('+')

        new_genotype = ''
        for level_1, level_2 in zip(levels1, levels2):
            level_1 = level_1.split('|')[1:-1]
            level_2 = level_2.split('|')[1:-1]
            n_genes = len(level_1)

            new_genotype += '|'
            for i in range(n_genes):
                choice = np.random.binomial(1, 0.5)
                if choice == 0:
                    new_genotype += level_1[i] + '|'
                else:
                    new_genotype += level_2[i] + '|'
            new_genotype += '+'

        index = self.get_index(new_genotype[:-1])
        return self[index]
