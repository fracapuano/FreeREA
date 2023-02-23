from pathlib import Path
from itertools import chain
from typing import List
import numpy as np

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