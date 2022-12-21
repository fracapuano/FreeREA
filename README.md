> **Disclaimer**
This notebook offers a tour of an implementation of [FreeREA: Training-Free Evolution-based Architecture Search](https://arxiv.org/pdf/2207.05135.pdf), by Cavagnero et al, 2022.
This work is not associated in any way or form with the authors and only aims at reproducing the findings presented by the authors in the cited paper. 
The main author has been informed of the publicity of this re-implementation. 


# Set-up

After having downloaded the repo, the first step to use this code is installing dependencies.

To set up dependancies, set up a virtual environment clonig the `env.yml` file.

```bash
cd freeREA && conda env create -f env.yml
```

Once dependencies are successfully installed, please go ahead and download **in the main folder** (that is, `freeREA`) the `archive` folder, containing the actual NATS-Bench networks. 

Moreover, archive also contains data that are otherwise re-downloaded. `archive` is already in the `.gitignore` file of this repo. 

To download `archive`, use [this link](https://drive.google.com/file/d/1LMpDiS1hmCLsC4Y86bhF41NzqAx5kS8c/view?usp=share_link) and then unzip-it.

# Examples
To see an example of how this repository can be used, please refer to [this notebook](https://github.com/fracapuano/freeREA/blob/main/FreeREA.ipynb).
