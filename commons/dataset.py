from .create_datasets import *

class DataSet: 
    def __init__(self, name:str="cifar10", batchsize:int=32): 
        self._name = name
        self._batchsize = batchsize

        self.accepted_datasets = ["cifar10", "cifar100", "imagenet16-120"]
    
    @property
    def name(self): 
        return self._name
    @name.setter
    def change(self, new_dataset:str):
        self._name = new_dataset.lower()
    @property
    def batchsize(self): 
        return self._batchsize
    @batchsize.setter
    def change_batchsize(self, new_batchsize:int): 
        if isinstance(new_batchsize, int) and new_batchsize <= 64:
            self._batchsize = new_batchsize
        else: 
            print("Friend don't friends use batchsizes larger than 64")
            raise ValueError("Batch size: {new_batchsize} not accepted")

    def set_trainloader(self): 
        if self._name in self.accepted_datasets:
            self.trainloader = cifar10(size=self.batchsize)[0]
        else: 
            raise NotImplementedError("Other datasets not yet implemented!")
    
    def set_testloader(self):
        if self._name in self.accepted_datasets: 
            self.testloader = cifar10(size=self.batchsize)[1]
        else: 
            raise NotImplementedError("Other datasets not yet implemented!") 

    def random_examples(self, loader:str="train", examples_only:bool=True)->Iterable[torch.Tensor]:
        """Return random examples from the dataset.
        
        Args: 
            loader (str, optional): Where to sample batches of data from.
                                    Either "train" or "test". Defaults to "train" 
            examples_only (bool, optional): Whether to return only examples (i.e., without
                                            returning corresponding labels). Defaults to True. 
        
        Returns: 
            Iterable[torch.Tensor]: Either examples only or an (examples, labels) iterable.
        """
        
        if loader.lower()=="train": 
            self.set_trainloader()
            if examples_only:  # returnig only examples
                return next(iter(self.trainloader))[0]
            else:  # returning examples and labels
                return next(self.trainloader)

        elif loader.lower()=="test":
            self.set_testloader()
            if examples_only: 
                return next(iter(self.testloader))[0]
            else: 
                return next(self.testloader)
        else:
            raise ValueError(f"Loader {loader} not accepted! Either 'train' or 'test'")
        
"""TODO: Implements similar functions for CIFAR-100 and ImageNet16-120"""