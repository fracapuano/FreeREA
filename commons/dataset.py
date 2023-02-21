from .create_datasets import *
from .create_datasets import name2dataset
from typing import Iterable
import torch

class Dataset: 
    def __init__(self, name:str="cifar10", batchsize:int=32): 
        self._name = name.lower()
        self._batchsize = batchsize
        self._traintest_builder = name2dataset(self._name, batch_size=self._batchsize, size=32 if self._name.startswith("cifar") else 16)

        self.accepted_datasets = ["cifar10", "cifar100", "imagenet16-120", "imagenet"]
                
    @property
    def name(self): 
        return self._name
    
    @name.setter
    def change_dataset(self, new_dataset:str):
        if new_dataset.lower in self.accepted_datasets:
            self._name = new_dataset.lower()
            self._traintest_builder = name2dataset(self._name, batch_size=self._batchsize, size=32 if self._name.startswith("cifar") else 16)

        else: 
            raise ValueError(f"{new_dataset} not in {self.accepted_datasets}")
    
    @property
    def batchsize(self): 
        return self._batchsize
    
    @batchsize.setter
    def change_batchsize(self, new_batchsize:int): 
        if isinstance(new_batchsize, int) and new_batchsize <= 64:
            self._batchsize = new_batchsize
        else: 
            print("Friends don't let friends use batchsizes larger than 64. \n\t (~Y. LeCunn)")
            raise ValueError(f"Batch size: {new_batchsize} not accepted")

    def set_trainloader(self):
        if self._name in self.accepted_datasets:
            self.trainloader, _ = self._traintest_builder
    
    def set_testloader(self):
        if self._name in self.accepted_datasets: 
            _, self.testloader = self._traintest_builder

    def random_examples(self, split:str="train", with_labels:bool=True)->Iterable[torch.Tensor]:
        """Return random examples from the dataset.
        
        Args: 
            split (str, optional): Where to sample batches of data from.
                                    Either "train" or "test". Defaults to "train" 
            with_labels (bool, optional): Whether to return labelled or un-labelled examples (i.e., without
                                            returning corresponding labels). Defaults to True (labelled examples).
        
        Returns: 
            Iterable[torch.Tensor]: Either examples only or an (examples, labels) iterable.
        """
        # access `train` split
        if split.lower()=="train": 
            self.set_trainloader()
            if with_labels:  # returnig examples and labels
                return next(iter(self.trainloader))
            else:  # returning examples only
                return next(iter(self.trainloader))[0]
        
        # access `test` split
        elif split.lower()=="test":
            self.set_testloader()
            if with_labels: # returnig examples and labels
                return next(iter(self.testloader))
            else: # returning examples only
                return next(iter(self.testloader))[0]

        else:
            raise ValueError(f"Split {split} not accepted! Either 'train' or 'test'")

