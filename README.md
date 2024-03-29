# Modeling dataset bias in machine-learned theories of economic decision making
This is the official repository, containing code and data for the paper:\
T. Thomas, D. Straub, F. Tatai, M. Shene, T. Tosik, K. Kersting, C. A. Rothkopf. Modelling dataset bias in machine-learned theories of economic decision-making. Nature Human Behaviour (2024)\
https://doi.org/10.1038/s41562-023-01784-6.

![Explanatory Figure](figure1.png)

## Data
All three datasets, i.e. [CPC15](https://economics.agri.huji.ac.il/crc2015/raw-data), [CPC18](https://cpc-18.com/data/) and [choices13k](https://github.com/jcpeterson/choices13k) used in this study have been publicly avialable.
Aggregate versions without individual data and with additional features and some utility columns are stored in the [data folder](./data/).

## Analysis
The [Analysis notebook](./Analysis.ipynb) shows how to reproduce many of the plots and analysis done in the paper.

## Models
Pretrained models under the name that they were shown in Table 1 in the paper are stored in the [models folder](./models).  
The [models notebook](./NNs.ipynb) gives simple examples how to load, save and train the NN models discussed in the paper.
The underlying source code for them is in the [src folder](./src).

## Usage
To use the notebooks and reproduce our results, install the conda environment, using
```
conda env create -f environment.yml
conda activate DecisionMaking
```

## Citation
Thomas, T., Straub, D., Tatai, F. et al. Modelling dataset bias in machine-learned theories of economic decision-making. Nat Hum Behav (2024). https://doi.org/10.1038/s41562-023-01784-6
