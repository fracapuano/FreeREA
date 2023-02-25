"""Pre-computes the metrics in `metrics` for each candidate architecture in the considered search-space."""
from commons import *
from metrics import *
from tqdm import tqdm
from multiprocessing import Pool
from typing import Text, List
import argparse
from scipy.stats import kendalltau, spearmanr
from itertools import product

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to be used in computing the considered metrics")
    parser.add_argument("--path-to-save", default="cachedmetrics", type=str, help="Where to save the cached-metrics files")

    return parser.parse_args()

args = parse_args()

this_dataset = args.dataset
cachedmetrics_path = args.path_to_save

def score_all_nets(dataset:str="cifar10", metrics:list=all_metrics, path_to_save:str="cachedmetrics")->None: 
    """Score all networks inside NATS-Bench for the same set of three metrics"""
    nats = NATSInterface(dataset=dataset)
    result = np.zeros(shape=(len(nats), 1+len(metrics)))
    # taking a random batch of images
    images = Dataset(name=dataset, batchsize=64).random_examples(with_labels=False)
    p_bar = tqdm(nats)
    p_bar.set_description(f"Dataset: {dataset}")

    for net_idx, net in enumerate(p_bar):
        # storing the metrics
        result[net_idx, :] = [net_idx] + [
            metric(net=metric_interface(metric, net), inputs=images) for metric in metrics
            ]
        
    np.savetxt(f"{path_to_save}/{dataset}_cachedmetrics.txt", result, header="Arch_Idx, NASWOT, logSynflow, PortionSkipped")

def performance_all_nets_dataset(dataset:str, training_epochs:int=200, path_to_save:str="cachedmetrics")->None:
    """Retrieves and saves the test accuracy, training time (both per epoch and total) of all networks given a considering dataset.
    
    Args: 
        dataset (str): Dataset to instantiate the API for.
        training_epochs (int, optional): Value of training epochs considered.
        path_to_save (str, optional): Whether to save the cached metrics used.
    
    Raises:
        ValueError: training_epochs is either 12 or 200.
    
    Returns: 
        None.
    """
    if training_epochs not in [12, 200]: 
        raise ValueError(f"Training Epochs accuracy logged only for 12 or 200 epochs! Prompted {training_epochs}")
    
    api = NATSInterface(dataset=dataset)
    results = np.zeros((len(api), 4))  # index column + accuracy, epoch time, total time for training

    pbar = tqdm(range(len(api)))
    pbar.set_description(f"Iterating over all datasets")
    for idx in pbar:
        results[idx, :] = [
            idx,
            api.query_test_performance(architecture_idx=idx, n_epochs=training_epochs)["accuracy"],  # test-accuracy
            api.query_training_performance(architecture_idx=idx, n_epochs=training_epochs)["per-epoch_time"],  # training per-epoch training time
            api.query_training_performance(architecture_idx=idx, n_epochs=training_epochs)["total_time"]  # training all-epochs training time
        ]

    np.savetxt(f"{path_to_save}/{dataset}_TrainTestMetrics.txt", results, header="ArchitectureIdx, Test-Accuracy, TrainingTime(xEpoch), TrainingTime(total)")
    print(f"{dataset}_TrainTestMetrics.txt saved at {path_to_save}.")


def performance_all_nets(datasets:List[Text]=["cifar10", "cifar100", "ImageNet16-120"])->None:
    """Returns a txt file with accuracies and training-stats for all networks in the API.
    
    Args: 
        datasets (List[Text], optional): list of all datasets considered for the usual benchmarks. Can also be reduced to some sub-element.
    """
    with Pool() as pool:
        pool.map(performance_all_nets_dataset, datasets)

def correlation(dataset:str, metric:str, corr_type:str="spearman", read:bool=True, cachedmetric:str="cachedmetrics", verbose:int=None)->float:
    """Computes corr_type correlation between a given metric and the test accuracy on a given dataset.

    Args:
        dataset (str): Dataset to consider for correlation.
        metric (str): Metric to be considered in computing correlation. Must be one of the ones in metrics_names.
        corr_type (str, optional): Correlation type.
        read (bool, optional): Whether or not to read the metric value instead of recomputing them. Defaults to True.
        cachedmetrics (str, optional): The folder where to find the cached-metrics. Defaults to cachedmetrics.
        verbose (int, optional). When >0, prints execution-related information. Defaults to None.
    
    Raises: 
        ValueError: when metric not in metrics_names (available on __init__ of metrics) or corr_type not in ["spearman", "kendall-tau"].

    Returns: 
        float: Correlation between the given metric and test accuracy.
    """
    metric = metric.lower()
    corr_type = corr_type.lower()

    if metric not in metrics_names:  # all metrics implemented so far
        raise ValueError(f"Metric {metric} not among implemented ones: {metrics_names}")
    if corr_type not in ["spearman", "kendall-tau"]:  # types of correlation considered
        raise ValueError(f"{corr_type} correlation not one of ['spearman', 'kendall-tau']")
    
    if not read: 
        raise NotImplementedError("Recomputing metrics from scratch not yet implemented")
    
    file_name = f"{cachedmetric}/{dataset}_cachedmetrics.txt"
    accuracy_file = f"{cachedmetric}/{dataset}_TrainTestMetrics.txt"
    metric_values = np.loadtxt(file_name)[:, 1+metrics_names.index(metric)]  # retrieving column associated with a given metric name
    metric_values[metric_values == -np.inf] = 0
    
    testacc_values = np.loadtxt(accuracy_file)[:, 1]  # accuracy is always at column index 1
    
    if corr_type == "kendall-tau":
        corr = kendalltau(x=metric_values, y=testacc_values).statistic
        if verbose:
            print(f"Kendal-Tau correlation ({metric.upper()}vsTESTACCURACY): "+"{:.4f}".format(corr))
        return corr

    elif corr_type == "spearman":
        corr = spearmanr(a=metric_values, b=testacc_values).statistic
        if verbose:
            print(f"Spearman-R correlation ({metric.upper()}vsTESTACCURACY): "+"{:.4f}".format(corr))
        return corr

def obtain_correlations(
    datasets:List[Text] = ["cifar10", "cifar100", "ImageNet16-120"], 
    metrics:List[Text] = ["naswot", "log-synflow", "portion-skipped-layers"], 
    corr_types:List[Text] = ["spearman", "kendall-tau"]
    )->None:
    """
    Computes correlation values for various combinations of datasets, correlation type and metrics considered. Correlation of metrics is 
    always mesured with respect to test accuracy.
    """
    for dataset, metric, corr_type in tqdm(product(datasets, metrics, corr_types)):
        c = correlation(dataset=dataset, metric=metric, corr_type=corr_type)
        outline = f"On {dataset} {metric} has a {corr_type}-correlation of " + "{:.4f}".format(c)
        print(outline)

def main():
    # obtain correlations between metrics defined in metrics/__init__.py and test accuracy.
    obtain_correlations()


if __name__ == "__main__":
    main()

