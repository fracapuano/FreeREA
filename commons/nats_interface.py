from nats_bench import create
from .utils import *
from xautodl.models import get_cell_based_tiny_net
from xautodl.models.cell_infers.tiny_network import TinyNetwork
from typing import Union, Tuple

class NATSInterface:
    def __init__(
        self, 
        path:str=str(get_project_root()) + "/archive/NATS-tss-v1_0-03ffb9-simple/",
        dataset:str="cifar10", 
        verbose:bool=False
        ):
    
        self._api = create(file_path_or_dict=path, search_space="topology", fast_mode=True, verbose=verbose)
        # sanity check on the given dataset
        self.NATS_datasets = ["cifar10", "cifar100", "imagenet16-120"]
        if dataset.lower() not in self.NATS_datasets: 
            raise ValueError(f"Dataset '{dataset}' not in {self.NATS_datasets}!")
        
        self._dataset = dataset

    @property
    def dataset(self): 
        return self._dataset
    
    @dataset.setter
    def change_dataset(self, new_dataset:str): 
        """Updates the current dataset with a new one"""
        if new_dataset.lower() in self.NATS_datasets: 
            self._dataset = new_dataset
        else: 
            raise ValueError(f"New dataset {new_dataset} not in {self.NATS_datasets}")
    
    def __len__(self):
        return len(self._api)
    
    def __getitem__(self, idx:int) -> TinyNetwork: 
        """Returns untrained network corresponding to index `idx`"""
        self.query_index(idx=idx, trained_weights=False)
    
    def nats_ops(self):
        return {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'avg_pool_3x3'}
    
    def query_with_index(
        self, 
        idx:int, 
        trained_weights:bool=False, 
        return_cell_structure:bool=False) -> Union[TinyNetwork, str]: 
        """This function returns the TinyNetwork object asssociated to index `idx`. The returned network
        is either trained or not with respect to `trained_weights`.

        Args:
            idx (int): Numerical index of the network to be returned.
            trained_weigths (bool, optional): Whether or not to load the state_dict for the returned network. Defaults to False.
            return_cell_structure (bool, optional): Whether or not to return the cell-structure for the considered network. Defaults to False.
        
        Returns:
            Union[TinyNetwork, str]: Either untrained or trained network corresponding to index idx. Optionally, the
                                      string representing the network cell structure is returned too.
        """
        net_config = self._api.get_net_config(index=idx, dataset=self.dataset)
        tinynet = get_cell_based_tiny_net(config=net_config)

        if return_cell_structure:
            if trained_weights: 
                # dictionary in which the key is the random seed for training and the values are the parameters                                                         
                params = self._api.get_net_param(index=idx, dataset=self._dataset, seed=None)
                return tinynet.load_state_dict(next(iter(params.values()))), net_config["arch_str"]
            else: 
                return tinynet, net_config["arch_str"] # untrained network
        else: 
            if trained_weights:                                                       
                params = self._api.get_net_param(index=idx, dataset=self._dataset, seed=None)
                return tinynet.load_state_dict(next(iter(params.values()))) 
            else: 
                return tinynet, net_config["arch_str"] # untrained network
    
    def query_with_architecture(
        self, 
        architecture_string:str, 
        trained_weights:bool=False, 
        return_cell_structure:bool=True) -> Union[TinyNetwork, list]: 
        """This function returns the TinyNetwork object associated to `architecture_string` architecture. The returned network
        is either trained or not with respect to `trained_weights`.

        Args:
            architecture_string (str): String representing a given architecture.
            trained_weigths (bool, optional): Whether or not to load the state_dict for the returned network. Defaults to False.
            return_cell_structure (bool, optional): Whether or not to return the cell-structure for the considered network. Defaults to False.

        Returns:
            TinyNetwork: Either untrained or trained network corresponding to index idx.
        """
        if not cellstructure_isvalid(input_str=architecture_string): 
            raise ValueError(f"Architecture {architecture_string} is not valid in NATS search space!")

        architecture_idx = self._api.query_index_by_arch(arch=architecture_string)
        net_config = self._api.get_net_config(index=architecture_idx, dataset=self._dataset)
        tinynet = get_cell_based_tiny_net(config=net_config)

        if return_cell_structure:
            if trained_weights: 
                # dictionary in which the key is the random seed for training and the values are the parameters                                                         
                params = self._api.get_net_param(index=architecture_idx, dataset=self._dataset, seed=None)  # must specify `seed=None`
                return tinynet.load_state_dict(next(iter(params.values()))), net_config["arch_str"]
            else: 
                return tinynet, net_config["arch_str"]  # untrained network
        else: 
            if trained_weights:                                                     
                params = self._api.get_net_param(index=architecture_idx, dataset=self._dataset, seed=None)
                return tinynet.load_state_dict(next(iter(params.values())))
            else: 
                return tinynet
    
    def query(
        self, 
        input_query:Tuple[int, str], 
        trained_weights:bool=False, 
        return_cell_structure:bool=False) -> Union[TinyNetwork, List]: 
        """This function unified query with index and query with architecture in one single `query` method.
        
        Args: 
            input_query (Tuple[int, str]): Either an integer or a string indicating, respectively, an index for the
                                           considered search space or a given cell structure
            trained_weigths (bool, optional): Whether or not to load the state_dict for the returned network. Defaults to False.
            return_cell_structure (bool, optional): Whether or not to return the cell-structure for the considered network. Defaults to False.
        
        Returns:
            Union[TinyNetwork, str]: Either untrained or trained network corresponding to index idx. Optionally, the
                                      string representing the network cell structure is returned too.
        """
        if isinstance(input_query, int): 
            return self.query_with_index(
                idx=input_query,
                trained_weights=trained_weights, 
                return_cell_structure=return_cell_structure
                )

        elif isinstance(input_query, str): 
            return self.query_with_architecture(
                architecture_string=input_query, 
                trained_weights=trained_weights, 
                return_cell_structure=return_cell_structure
                )
        else: 
            raise ValueError("{:} is not a string or an index indicating an architecture in NATS bench".format(input_query))

    def query_training_performance(self, architecture_idx:int, n_epochs:Tuple[int, int]=200) -> dict:
        """Returns accuracy, per-epoch and for n_epochs time for the training process of architecture `architecture_idx`"""
        result = dict()
        metrics = self._api.query_meta_info_by_index(
            arch_index=architecture_idx, 
            hp=str(n_epochs)
        ).get_metrics(dataset=self._dataset, setname="train")
        
        # only storing some of the metrics saved in the architecture
        result["accuracy"] = metrics["accuracy"]
        result["per-epoch_time"] = metrics["cur_time"]
        result["total_time"] = metrics["all_time"]

        return result
    
    def query_test_performance(
        self, 
        architecture_idx:int, 
        n_epochs:Tuple[int, int]=200
        ) -> dict:
        """Returns accuracy, per-epoch and for n_epochs time related to testing of architecture `architecture_idx`"""
        result = dict()
        metrics = self._api.query_meta_info_by_index(
            arch_index=architecture_idx, 
            hp=str(n_epochs)
        ).get_metrics(dataset=self._dataset, setname="ori-test")
        
        # only storing some of the metrics saved in the architecture
        result["accuracy"] = metrics["accuracy"]
        result["per-epoch_time"] = metrics["cur_time"]
        result["total_time"] = metrics["all_time"]

        return result
    
    def generate_random_samples(
        self, 
        n_samples:int=10) -> Tuple[List, List]:
        """Generate a group of architectures chosen at random"""
        idxs = [self._api.random() for _ in range(n_samples)]
        tinynets = [self.query_with_index(i) for i in idxs]
        cell_structures = [self._api.get_net_config(index=i, dataset=self.dataset)["arch_str"] for i in idxs]
        # return tinynets and cell_structures_string
        return tinynets, cell_structures
