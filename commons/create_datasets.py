"""Defines various dataset builders (mainly inspired to: https://github.com/NiccoloCavagnero/FreeREA/blob/master/datasets.py)"""
import torchvision
from torch.utils.data import DataLoader
from .utils import get_project_root
from xautodl.datasets.DownsampledImageNet import ImageNet16
import torchvision.transforms as transforms
from typing import Tuple

# transformations to perform data-augmentation in CIFAR-like train datasets.
def cifar_transforms(train:bool=True, size:int=32)->transforms.Compose:
    """Returns transformation.Compose object for train/test CIFAR-like dataset.
    
    Args:
        train (bool, optional): Whether or not to return tranformations to be applied to the training dataset. 
                                Defaults to True.
        size (int, optional): Size of the images to be used in both training and testing datasets. Defaults to 32.

    Returns:
        transforms.Compose: Composition of all tranformations needed to manage input data.
    """
    if train: # transformations to be applied on training dataset
        cifar_train_transforms = transforms.Compose(
                [
                transforms.RandomCrop(size=size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            )
        return cifar_train_transforms

    elif not train: # transformations to be applied on test dataset
        cifar_test_transforms = transforms.Compose(
            [
            *((transforms.Resize(size),) if size != 32 else ()),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )
        return cifar_test_transforms


def imagenet_transform(train:bool=True, size:int=16)->transforms.Compose:
    """Returns transformation.Compose object for train/test ImageNet16-120 dataset.
    Args:
        train (bool, optional): Whether or not to return tranformations to be applied to the training dataset. 
                                Defaults to True.
        size (int, optional): Size of the images to be used in both training and testing datasets. Defaults to 16.

    Returns:
        transforms.Compose: Composition of all tranformations needed to manage input data.
    """
    if train: # transformations to be applied on training dataset
        imagenet_train_transforms = transforms.Compose(
            [
            transforms.RandomCrop(size, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4810980392156863, 0.45749019607843133, 0.4078823529411765),
                (0.247921568627451, 0.24023529411764705, 0.2552549019607843)
                )
            ]
        )
        return imagenet_train_transforms

    elif not train: # transformations to be applied on test dataset
        imagenet_test_transforms = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4810980392156863, 0.45749019607843133, 0.4078823529411765),
                (0.247921568627451, 0.24023529411764705, 0.2552549019607843)
                )
            ]
        )
        return imagenet_test_transforms


def cifar10(path:str=str(get_project_root()) + "/archive/data", size:int=32, batch_size:int=32)->Tuple[DataLoader, DataLoader]:
    """Returns train/test DataLoaders for the cifar10 dataset.
    
    Args: 
        path (str, optional): Where to find (if yet present) or download the CIFAR10 dataset. Defaults to (archive/data)
        size (int, optional): Size of the images to be used in the transformations. Defaults to 32.
        batch_size (int, optional): Batch-size for the DataLoader object. Defaults to 32. 
    
    Returns: 
        Tuple[DataLoader, DataLoader]: Tuple of DataLoader object in which the first element is the train-set dataloader and 
                                       second element is test-set dataloader.
    """

    # define a training set in the path folder
    is_training = True
    trainset = torchvision.datasets.CIFAR10(root=path, 
                                            train=is_training,
                                            download=True, 
                                            transform=cifar_transforms(train=is_training, size=size)
                                            )
    # define trainingset DataLoader
    trainloader = DataLoader(trainset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            )
    # define a test set in the path folder
    is_training = False
    testset = torchvision.datasets.CIFAR10(root=path, 
                                           train=is_training,
                                           download=True, 
                                           transform=cifar_transforms(train=is_training, size=size)
                                           )
    # define trainingset loader
    testloader = DataLoader(testset, 
                            batch_size=batch_size,
                            shuffle=False
                            )
    # return an iterable of DataLoaders
    return [trainloader, testloader]


def cifar100(path:str=str(get_project_root()) + "/archive/data", size:int=32, batch_size:int=32)->Tuple[DataLoader, DataLoader]:
    """Returns train/test DataLoaders for the cifar100 dataset.
    
    Args: 
        path (str, optional): Where to find (if yet present) or download the CIFAR100 dataset. Defaults to (archive/data)
        size (int, optional): Size of the images to be used in the transformations. Defaults to 32.
        batch_size (int, optional): Batch-size for the DataLoader object. Defaults to 32. 
    
    Returns: 
        Tuple[DataLoader, DataLoader]: Tuple of DataLoader object in which the first element is the train-set dataloader and 
                                       second element is test-set dataloader.
    """

    # define a training set in the path folder
    is_training = True
    trainset = torchvision.datasets.CIFAR100(root=path, 
                                            train=is_training,
                                            download=True, 
                                            transform=cifar_transforms(train=is_training, size=size)
                                            )
    # define trainingset DataLoader
    trainloader = DataLoader(trainset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            )
    # define a test set in the path folder
    is_training = False
    testset = torchvision.datasets.CIFAR100(root=path, 
                                           train=is_training,
                                           download=True, 
                                           transform=cifar_transforms(train=is_training, size=size)
                                           )
    # define trainingset loader
    testloader = DataLoader(testset, 
                            batch_size=batch_size,
                            shuffle=False
                            )
    # return an iterable of DataLoaders
    return [trainloader, testloader]


def imagenet16_120(path:str=str(get_project_root()) + "/archive/data", size:int=16, batch_size:int=32)->Tuple[DataLoader, DataLoader]:
    """Returns train/test DataLoaders for the ImageNet16-120 dataset. 
    ImageNet16-120 is a subsample of the ImageNet dataset containing 16x16 pixel images of 120 classes only (for more details
    about ImageNet16-120 see https://arxiv.org/pdf/1707.08819.pdf).

    Args:
        path (str, optional):  Where to find (if yet present) or download the ImageNet16-120 dataset. Defaults to str(get_project_root())+"/archive/data".
        size (int, optional): Size of the images to be used in the transformations. Mainly added for compatibility. Defaults to 16.
        batch_size (int, optional): Batch-size for the DataLoader object. Defaults to 32.

    Raises: 
        ValueError: When size!=16. ImageNet16-120 only admits a value of 16 for the images size.

    Returns:
        Tuple[DataLoader, DataLoader]: _description_
    """
    # "size" argument is here just for compatibility
    if size!=16: 
        raise ValueError(f"ImageNet16 is only for 16x16 images! Prompted {size}x{size}.")
    
    is_training = True
    # define a training set in the path folder
    trainset = ImageNet16(root=path, 
                          train=is_training, 
                          transform=imagenet_transform(train=is_training, size=size),
                          use_num_of_class_only=120)
    # define trainingset DataLoader
    trainloader = DataLoader(trainset,
                             batch_size=batch_size, 
                             shuffle=True)
    
    is_training = False
    # define a test set in the path folder
    testset = ImageNet16(root=path, 
                         train=is_training, 
                         transform=imagenet_transform(train=is_training, size=size),
                         use_num_of_class_only=120)
    # define trainingset loader
    testloader = DataLoader(testset, 
                            batch_size=batch_size,
                            shuffle=False)

    return [trainloader, testloader]

name2dataset = {
    "cifar10": cifar10(),
    "cifar100": cifar100(), 
    "imagenet16-120": imagenet16_120(),
    "imagenet": imagenet16_120()
}

