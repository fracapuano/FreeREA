*Edit: The authors official implementation is now available on GitHub at [github.com/NiccoloCavagnero/FreeREA](https://github.com/NiccoloCavagnero/FreeREA).*

> **Disclaimer**
This repo presents an implementation of [FreeREA: Training-Free Evolution-based Architecture Search](https://arxiv.org/pdf/2207.05135.pdf), by Cavagnero et al, 2022.
This work is not associated in any way or form with the authors and only aims at reproducing the findings presented by the authors in the cited paper. 
The authors have been informed of this re-implementation.

# Set-up
After having locally cloned this repo, the first step to use this code is installing the required dependencies.
To set up dependancies, set up a virtual environment clonig the `env.yml` file.

```bash
$ conda create --name <env_name> --file requirements.txt
```

Once dependencies have been successfully installed, please go ahead and download **in the main folder** (that is, `FreeREA`) the `archive` folder, containing the actual NATS-Bench networks. `archive` is already in the `.gitignore` file of this repo.

To download NATS-Bench and create the `archive` folder simply run the following:
```bash
$ bash setup_nats.sh
```
Alternatively, one could download `archive` from [here](https://drive.google.com/file/d/1LMpDiS1hmCLsC4Y86bhF41NzqAx5kS8c/view) and then unzip the folder. 

Please consider that downloading the search space only is more than sufficient as fully trained models are not needed, since the benchmark conveniently stores the model performance metrics. More than that, downloading the trained architectures (that is, the fully trained architectures with their weights) would download 200+ GB of architectures.