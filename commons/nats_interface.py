from nats_bench import create
from .utils import *
from xautodl.models import get_cell_based_tiny_net
from xautodl.models.cell_infers.tiny_network import TinyNetwork
from typing import Union, Tuple
import numpy as np
import json 

def jsonKeys2int(x):
    """Function to convert the keys of a dictionary back to integers after having used json module
    More here: https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
    """
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

class NATSInterface:
    def __init__(
        self, 
        dataset:str="cifar10", 
        ):
    
        with open('checkpoint_1.json', 'rb') as f:
            self._api = jsonKeys2int(json.load(f))
        # sanity check on the given dataset
        self.NATS_datasets = ["cifar10", "cifar100", "ImageNet16-120"]
        if dataset.lower() not in self.NATS_datasets: 
            if 'imagenet' in dataset.lower():
                dataset = 'ImageNet16-120'
            else:
                raise ValueError(f"Dataset '{dataset}' not in {self.NATS_datasets}!")
        
        self._dataset = dataset

    @property
    def dataset(self): 
        return self._dataset
    
    @dataset.setter
    def change_dataset(self, new_dataset:str): 
        """
        Updates the current dataset with a new one. 
        Raises ValueError when new_dataset is not one of ["cifar10", "cifar100", "imagenet16-120"]
        """
        if new_dataset.lower() in self.NATS_datasets: 
            self._dataset = new_dataset
        else: 
            raise ValueError(f"New dataset {new_dataset} not in {self.NATS_datasets}")
    
    def __len__(self):
        return len(self._api.keys())
    
    def __getitem__(self, idx:int) -> TinyNetwork: 
        """Returns (untrained) network corresponding to index `idx`"""
        return self.query_with_index(idx=idx)

    def __iter__(self):
        """Iterator method"""
        self.iteration_index = 0
        return self

    def __next__(self):
        if self.iteration_index >= self.__len__():
            raise StopIteration
        # access current element 
        net = self[self.iteration_index]
        # update the iteration index
        self.iteration_index += 1
        return net
    
    def nats_ops(self):
        return {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3'}
    
    def query_with_index(
        self, 
        idx:int) -> str: 
        """This function returns the architecture string associated to this index
        """
        return self._api[idx]['architecture_string']
    
    def query_with_architecture(
        self, 
        architecture_string:str) -> int: 
        """This function returns the index idx associated to `architecture_string` architecture
        """
        return [k for k, v in self._api.items() if v['architecture_string'] == architecture_string][0]
   
    def query(
        self, 
        input_query:Tuple[int, str]): 
        """This function unified query with index and query with architecture in one single `query` method.
        """
        if isinstance(input_query, int): 
            return self.query_with_index(
                idx=input_query
                )

        elif isinstance(input_query, str): 
            return self.query_with_architecture(
                architecture_string=input_query
                )
        else: 
            raise ValueError("{:} is not a string or an index indicating an architecture in NATS bench".format(input_query))

    # def query_training_performance(self, architecture_idx:int, n_epochs:Tuple[int, int]=200) -> dict:
    #     """Returns accuracy, per-epoch and for n_epochs time for the training process of architecture `architecture_idx`"""
    #     result = dict()
    #     metrics = self._api.query_meta_info_by_index(
    #         arch_index=architecture_idx, 
    #         hp=str(n_epochs)
    #     ).get_metrics(dataset=self._dataset, setname="train")
        
    #     # only storing some of the metrics saved in the architecture
    #     result["accuracy"] = metrics["accuracy"]
    #     result["per-epoch_time"] = metrics["cur_time"]
    #     result["total_time"] = metrics["all_time"]

    #     return result
    
    def query_test_performance(
        self, 
        architecture_idx:int
        ) -> float:
        """Returns test accuracy for architecture `architecture_idx`"""

        return self._api[architecture_idx][self.dataset]['test_accuracy']
    
    def generate_random_samples(
        self, 
        n_samples:int=10) -> Tuple[List, List]:
        """Generate a group of architectures chosen at random"""
        idxs = np.random.choice(self.__len__(), size=n_samples, replace=False)
        arch_strings = [self._api[idx]['architecture_string'] for idx in idxs]
        # return arch_strings and the unique indices of the networks
        return arch_strings, idxs
