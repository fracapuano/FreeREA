from commons.utils import * 

def compute_skipped_layers(genotype:list) -> float: 
    """Computes the portion of skipped layers wiht respect to the number of skip connections.
    This metric is a proxy for how many skip connections are in place and, since these very metrics 
    are at the core of trainability, this metric serves as a measure of trainability too. 
    
    Args: 
        genotype (list): Genotype of the considered invididual. In FreeREA the genotype coincides with the
                         list of all the operations in each cell.
    
    Returns: 
        float: Portion of skipped layers over number of skip connections
    """
    # turn the genotype list into original string - done to reconstruct intra-level relationships
    genotype_str = genotype_to_architecture(genotype=genotype)
    # split levels
    levels = genotype_str.split('+')
    max_len = 0
    counter = 0
    # cycle over all levels
    for idx, level in enumerate(levels):
        level = level.split('|')[1:-1]
        n_genes = len(level)
        # cycle over connections between levels
        for i in range(n_genes):
            if 'skip' in level[i]:
                counter += 1
                min_edge = idx - i
                max_len += min_edge  # counts number of skipped layers 
    if counter:
        return max_len / counter  # portion of skipped layers over all possible skip connections
    return 0