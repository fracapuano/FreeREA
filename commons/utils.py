from pathlib import Path
from itertools import chain
from typing import List

def get_project_root(): 
    """
    Returns project root directory from this script nested in the commons folder.
    """
    return Path(__file__).parent.parent

def check_architecture(input_str:str)->bool: 
    """Checks if the format of a given architecture is valid for the NATS Bench topology space"""
    all_ops = {'none', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1'}
    all_numbers = {'0', '1', '2'}

    subcells = input_str.split("+")  # divide the input string into different levels
    ops = chain(*[subcell.split("|")[1:-1] for subcell in subcells])  # divide into different nodes to retrieve ops
    subops = chain(*[op.split("~") for op in ops]) # divide into operation and node

    is_valid = all([(n in all_ops) or (n in all_numbers) for n in subops]) # check if the full string is valid
    return is_valid

def architecture_to_genotype(arch_str:str)->List: 
    """Turn architectures string into genotype list"""
    all_ops = {'none', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1'}
    all_numbers = {'0', '1', '2'}

    subcells = arch_str.split("+")  # divide the input string into different levels
    ops = chain(*[subcell.split("|")[1:-1] for subcell in subcells])  # divide into different nodes to retrieve ops
    return list(ops)

def genotype_to_architecture(genotype:List)->str: 
    """Reformats genotype as architecture string"""
    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*genotype)

def genotype_is_valid(genotype:List)->bool:
    """Checks whether or not genotype is valid"""
    all_ops = {'none', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1'}
    all_numbers = {'0', '1', '2'}

    subops = chain(*[op.split("~") for op in genotype]) # divide into operation and node
    is_valid = all([(n in all_ops) or (n in all_numbers) for n in subops]) # check if the full string is valid
    return is_valid

