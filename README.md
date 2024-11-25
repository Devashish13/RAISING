# RAISING

This repository contains the source code, simulation data, and documentation for the **RAISING**, a two-stage deep learning framework that first performs hyperparameter tuning to devise the best NN architecture and then performs training on the entire data to estimate the feature importance. The method has been published in Nucleic Acids Research (<https://doi.org/10.1093/nar/gkae1027>). The source code is available on GitHub(<https://github.com/Devashish13/RAISING/>) 

# Abstract
Response to spatiotemporal variation in selection gradients resulted in signatures of polygenic adaptation in human genomes. We introduce RAISING, a two-stage deep learning framework that optimizes neural network architecture through hyperparameter tuning before performing feature selection and prediction tasks. We tested RAISING on published and newly designed simulations that incorporate the complex interplay between demographic history and selection gradients. RAISING outperformed Phylogenetic Generalized Least Squares (PGLS), ridge regression and DeepGenomeScan, with significantly higher true positive rates (TPR) in detecting genetic adaptation. It reduced computational time by 60-fold and increased TPR by up to 28% compared to DeepGenomeScan on published data. In more complex demographic simulations, RAISING showed lower false discoveries and significantly higher TPR, up to 17-fold, compared to other methods. RAISING demonstrated robustness with least sensitivity to demographic history, selection gradient and their interactions.

## RAISING installation

Create a conda environment to install the RAISING

```
conda create -n RAISING_env python=3.9
conda activate RAISING_env
```
Install the package through pypi
```
pip install RAISING
```

Install the package through the github repository

```
pip install git+https://github.com/Devashish13/RAISING.git
```

## RAISING implementation 
Please visit the following link for detailed description of RAISING documentation and tutorials

<https://devashish13.github.io/RAISING/>

## Simulated data generated for detecting polygenic adaptation
Simulated data generated in this study for detecting polygenic selection can be accessed from zenodo repository(<https://zenodo.org/records/12752105>)
