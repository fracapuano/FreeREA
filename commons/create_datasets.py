import torch
import torchvision
from torch.utils.data import DataLoader
from .utils import get_project_root
import torchvision.transforms as transforms
from typing import Iterable

def cifar10(path:str=str(get_project_root()) + "/archive/data", size:int=32)->Iterable[DataLoader]:
    """Returns train/test DataLoaders for the cifar10 dataset.
    
    Args: 
        path (str, optional): Where to find (if yet present) or download the CIFAR10 dataset. Defaults to (archive/data)
        size (int, optional)"""

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )

    batch_size = size
    # define a training set in the path folder
    trainset = torchvision.datasets.CIFAR10(root=path, 
                                            train=True,
                                            download=True, 
                                            transform=transform
                                            )
    # define trainingset loader
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                             )
    # define a test set in the path folder
    testset = torchvision.datasets.CIFAR10(root=path, 
                                           train=False,
                                           download=True, 
                                           transform=transform
                                           )
    # define trainingset loader
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=batch_size,
                                             shuffle=False
                                             )
    # return an iterable of DataLoaders
    return [trainloader, testloader]

"""TODO: Implements similar functions for CIFAR-100 and ImageNet16-120"""
