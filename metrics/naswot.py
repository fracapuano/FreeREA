import torch
import torch.nn as nn
from commons.utils import correlation

def compute_naswot(
    net: nn.Module, 
    inputs: torch.Tensor, 
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    method: str = 'logdet') -> float:
    """Computes the NASWOT score for a given network
    
    Args: 
        net (nn.Module): Actual network to be scored according to naswot.
        inputs (torch.Tensor): Tensor of size `batch_size` corresponding to the images forwarded as input.
        device (torch.device): Either CPU or GPU device.
        method (str): one between 'logdet' and 'corr'. Defaults to 'logdet'
    """
    if method not in ['logdet', 'corr']:
        raise ValueError('Method not implement. Please pick one between logdet and corr')
    # gradients are completely useless here
    with torch.no_grad():
        # result of hooks
        cs = list()

        def naswot_hook(module: nn.Module, module_input: torch.Tensor, module_output: torch.Tensor) -> None:
            """
            This function hooks an extra-operation to forward pass of module `m`.

            Args:
                module (nn.Module): layer in neural network
                module_input (torch.Tensor): input to the considered module. Size dependent on the actual module.
                module_output (torch.Tensor): output to the considered module. Size dependendant on the actual module.
            """
            code = (module_output > 0).flatten(start_dim=1)  # binarize output to True/False whether not zero or zero.
            cs.append(code)  # store embedding

        # storing applied hooks to remove them from array when they are not needed anymore
        hooks = list()
        for m in net.modules():
            if isinstance(m, nn.ReLU):  # naswot is defined for ReLU only layers
                hooks.append(m.register_forward_hook(naswot_hook))  # register naswot hook for ReLU layers

        net.double().to(device)
        inputs = inputs.to(device)
        # populating cs with the ReLU embeddings
        _ = net(inputs.double())

        # removing hooks once they have been used
        for h in hooks:
            h.remove()

        # False-True embedding of the whole network, discarding non ReLU layers
        full_code = torch.cat(cs, dim=1)

        # codes and network output not needed
        del cs, _
        # mapping False->0 / True->1
        full_code_float = full_code.float()
        # number of concordances between the embeddings of `inputs`
        k = full_code_float @ full_code_float.t()
        # not needed anymore
        del full_code_float
        # mapping each False->1 / True->0
        not_full_code_float = torch.logical_not(full_code).float()
        # hamming distance is number of disagreements = size of array - number of agreeements
        k += not_full_code_float @ not_full_code_float.t()
        # naswot score computed on k
        if method == 'logdet':
            naswot_score = torch.slogdet(k).logabsdet.item()
        elif method == 'corr':
            naswot_score = correlation(k)
            
        return naswot_score
