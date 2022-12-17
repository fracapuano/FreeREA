from pathlib import Path
import os
import re
import pickle
import numpy as np
import pandas as pd
import gzip
from itertools import chain


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


def load_metrics(dataset_metric_folder: str, include_accuracies: bool = False, include_times: bool = True):
    # List of CSV files
    paths = [os.path.join(dataset_metric_folder, f) for f in os.listdir(dataset_metric_folder)]

    # Load all of them into a DataFrame
    df = None

    for file_ in paths:
        # Do not include accuracy if not required
        if not re.search(r'accuracies\.(?:csv|pkl)(?:\.gz)?$', file_) or include_accuracies:
            # Read metric
            open_fn = open
            if file_.endswith('.gz'):
                open_fn = gzip.open

            with open_fn(file_, 'rb') as fp:
                if re.search(r'\.csv(?:\.gz)?$', file_):
                    other = pd.read_csv(fp)
                elif re.search(r'\.pkl(?:\.gz)?$', file_):
                    other = pickle.load(fp)
                else:
                    continue

            if len(other.columns) > 2:
                metric = other.columns[1]
                if include_times:
                    other.rename(columns={'time=[s]': metric + '_time=[s]'}, inplace=True)

                # Remove time 'pure' columns
                other.drop(columns=[cn for cn in other.columns if cn.startswith('time')], inplace=True)

            # Merge
            if df is not None:
                df = df.merge(other, on='index')
            else:
                df = other

    df.drop(columns='index', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.min(), inplace=True)

    return df