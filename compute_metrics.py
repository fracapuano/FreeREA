"""Pre-computes the metrics in `metrics` for each candidate architecture in the considered search-space."""
from commons import *
from metrics import *
from tqdm import tqdm

import time

def score_all_nets(dataset:str="cifar10", metrics:list=all_metrics, path_to_save:str="cachedmetrics"): 
    """Score all networks inside NATS-Bench for the same set of three metrics"""
    nats = NATSInterface(dataset=dataset)
    result = np.zeros(shape=(len(nats), 1+len(metrics)))
    # taking a random batch of images
    images = Dataset(name=dataset, batchsize=64).random_examples(with_labels=False)

    for net_idx, net in enumerate(tqdm(nats)):
        # storing the metrics
        result[net_idx, :] = [net_idx] + [
            metric(net=metric_interface(metric, net), inputs=images) for metric in metrics
            ]
        
    np.savetxt(f"{path_to_save}/{dataset}_cachedmetrics.txt", result, header="Arch_Idx, NASWOT, logSynflow, PortionSkipped")

def main():
    score_all_nets(dataset="ImageNet16-120")

if __name__ == "__main__":
    main()
