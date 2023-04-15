from .naswot import *
from .logsynflow import *
from .skipped_layers import *
from typing import Tuple, Callable
from commons.utils import architecture_to_genotype

def compatible_skipped_layers(net:str, inputs: torch.Tensor):
    """
    This function computes a metric related to the number of skipped layers. 
    The inputs and random_init rguments is used for compatibility only
    """
    return compute_skipped_layers(architecture_to_genotype(net))

metric_input_type = {
    compute_naswot: nn.Module, 
    compute_logsynflow: nn.Module,
    compatible_skipped_layers: str
}

def metric_interface(metric:Callable, net:tuple)->Tuple[nn.Module, str]:
    """Interfaces each metric function with tuples in NATS Interface"""
    if metric_input_type[metric] == nn.Module:
        return net[0]  # instance of the TinyNetwork network
    elif metric_input_type[metric] == str:
        return net[1]  # architecture string

all_metrics = [
    compute_naswot, 
    compute_logsynflow,
    compatible_skipped_layers
]

metrics_names = [
    "naswot", 
    "log-synflow",
    "portion-skipped-layers"
]