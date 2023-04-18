from pathlib import Path
from itertools import chain
from typing import List
import numpy as np
import pandas as pd
import random
import pickle
import torch
import torch.nn as nn

def load_images(
        dataset:str='cifar100', 
        batch_size:int=32, 
        with_labels:bool=False,
        verbose:int=None
        )->object:
    """TODO: Add documentation."""
    if dataset not in ["cifar10", "cifar100", "imagenet"]:
        if 'imagenet' not in dataset.lower():
            raise ValueError('Please specify a valid dataset. Should be one of cifar10, cifar100, ImageNet')
        else:
            dataset = 'imagenet'
    if batch_size not in [32, 64]:
            raise ValueError(f"Batch size: {batch_size} not accepted. Can only be 32 or 64.")
    # sampling one random batch randomly
    random_batch = random.randrange(10)

    with open(f'data/{dataset}__batch{batch_size}_{random_batch}', 'rb') as pickle_file:
        images = pickle.load(pickle_file)
        if verbose: 
            print(f'Batch #{random_batch} loaded.')

    # returning one of the random batches generated randomly
    if with_labels:
        return images  # returns labelled examples
    else: 
        return images[0].float()  # only returns data, with no labels. Imagenet tensors are in uint8 hence mapping to floats

def read_lookup_table(dataset:str="cifar100"):
    """
    Returns the lookup table for the corresponding dataset
    """
    if dataset not in ["cifar10", "cifar100", "ImageNet16-120"]:
        if 'imagenet' not in dataset.lower():
            raise ValueError('Please specify a valid dataset. Should be one of cifar10, cifar100, ImageNet')
        else:
            dataset = 'ImageNet16-120'
    lookup_table = np.loadtxt(f'cachedmetrics/{dataset}_cachedmetrics.txt', skiprows=1)
    return lookup_table

def read_test_metrics(dataset:str="cifar100"):
    """
    Returns the train/test metrics cached table for the corresponding dataset
    """
    if dataset not in ["cifar10", "cifar100", "ImageNet16-120"]:
        if 'imagenet' not in dataset.lower():
            raise ValueError('Please specify a valid dataset. Should be one of cifar10, cifar100, ImageNet')
        else:
            dataset = 'imagenet'
    lookup_table = np.loadtxt(f'cachedmetrics/{dataset}_perfmetrics.txt', skiprows=1)
    return lookup_table

def load_normalization(dataset:str, normalization:str):
    """Returns the parameters for the chosen normalization on the dataset

    Args:
        dataset (str): dataset in use. Should be one of 'cifar10', 'cifar100', 'ImageNet'.
        normalization (str): normalization to use. Should be one of 'minmax', 'standard'.

    Returns:
        pd.DataFrame: parameters for the normalization
    """
    if dataset not in ["cifar10", "cifar100", "ImageNet16-120"]:
        if 'imagenet' not in dataset.lower():
            raise ValueError('Please specify a valid dataset. Should be one of cifar10, cifar100, ImageNet')
        else:
            dataset = 'ImageNet16-120'
    
    return pd.read_csv(f'cachedmetrics/{dataset}_{normalization}.csv')

def get_project_root(): 
    """
    Returns project root directory from this script nested in the commons folder.
    """
    return Path(__file__).parent.parent

def architecture_to_genotype(arch_str:str)->List: 
    """Turn architectures string into genotype list
    
    Args: 
        arch_str(str): String characterising the cell structure only. 
    
    Returns: 
        List: List containing the operations in the input cell structure.
              In a genetic-algorithm setting, this description represents a genotype. 
    """
    all_ops = {'none', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1'}
    all_numbers = {'0', '1', '2'}

    subcells = arch_str.split("+")  # divide the input string into different levels
    ops = chain(*[subcell.split("|")[1:-1] for subcell in subcells])  # divide into different nodes to retrieve ops
    return list(ops)

def genotype_to_architecture(genotype:List)->str: 
    """Reformats genotype as architecture string"""
    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*genotype)

def cellstructure_isvalid(input_str:str)->bool: 
    """Checks if the format of a given cell structure is valid for the NATS Bench topology space"""
    
    all_ops = {'none', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1'}
    all_numbers = {'0', '1', '2'}

    subcells = input_str.split("+")  # divide the input string into different levels
    ops = chain(*[subcell.split("|")[1:-1] for subcell in subcells])  # divide into different nodes to retrieve ops
    subops = chain(*[op.split("~") for op in ops]) # divide into operation and node

    is_valid = all([(n in all_ops) or (n in all_numbers) for n in subops]) # check if the full string is valid
    return is_valid

def genotype_is_valid(genotype:List)->bool:
    """Checks whether or not genotype is valid for the NATS Bench topology space"""
    all_ops = {'none', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1'}
    all_numbers = {'0', '1', '2'}

    subops = chain(*[op.split("~") for op in genotype]) # divide into operation and node
    is_valid = all([(n in all_ops) or (n in all_numbers) for n in subops]) # check if the full string is valid
    return is_valid

def correlation(tensor:torch.tensor)->float:
    """Compute correlation coefficient on a tensor, based on
    https://math.stackexchange.com/a/1393907

    Args:
        tensor (torch.tensor):

    Returns:
        float: Pearson correlation coefficient
    """
    tensor = tensor.double()
    r1 = torch.tensor(range(1, tensor.shape[0] + 1)).double()
    r2 = torch.tensor([i*i for i in range(1, tensor.shape[0] + 1)]).double()
    j = torch.ones(tensor.shape[0]).double()
    n = torch.matmul(torch.matmul(j, tensor), j.T).double()
    x = torch.matmul(torch.matmul(r1, tensor), j.T)
    y = torch.matmul(torch.matmul(j, tensor), r1.T)
    x2 = torch.matmul(torch.matmul(r2, tensor), j.T)
    y2 = torch.matmul(torch.matmul(j, tensor), r2.T)
    xy = torch.matmul(torch.matmul(r1, tensor), r1.T)
    
    corr = (n * xy - x * y) / (torch.sqrt(n * x2 - x**2) * torch.sqrt(n * y2 - y**2))

    return corr.item()

def kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def init_model(model):
    """Applies kaiming normal weights initialization to input model."""
    model.apply(kaiming_normal)
    return model