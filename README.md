![image](https://github.com/EthoML/VAME/assets/844306/0f08424f-06ab-48e4-8094-da0f0c78a08d)

🌟 Welcome to EthoML/VAME (Variational Animal Motion Encoding), an open-source machine learning tool for behavioral segmentation and analyses.

We are a group of behavioral enthusiasts, comprising the original VAME developers Kevin Luxem and Pavol Bauer, behavioral neuroscientists Stephanie R. Miller and Jorge J. Palop, and computer scientists and statisticians Alex Pico, Reuben Thomas, and Katie Ly). Our aim is to provide scalable, unbiased and sensitive approaches for assessing mouse behavior using computer vision and machine learning approaches.

We are focused on the expanding the analytical capabilities of VAME segmentation by providing curated scripts for VAME implementation and tools for data processing, visualization, and statistical analyses. 

## Recent Improvements to VAME
* Curated scripts for VAME implementation
* Addition of a new cost function for community dendrogram generation
* Addition of a new egocentric alignment method
* Addition of hardware instructions for video capturing of mouse behavior.
* Addition of mouse behavioral videos for practicing VAME and for benchmarking purposes
* Refined output filename structure

[Insert open field videos]


## Authors and Code Contributors
VAME was developed by Kevin Luxem and Pavol Bauer (Luxem et. al., 2022). The original VAME repository was deprecated, forked, and is now being maintained here at https://github.com/EthoML/VAME.

The development of VAME is heavily inspired by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/). As such, the VAME project management codebase has been adapted from the DeepLabCut codebase. The DeepLabCut 2.0 toolbox is © A. & M.W. Mathis Labs [deeplabcut.org](http:\\deeplabcut.org), released under LGPL v3.0. The implementation of the VRAE model is partially adapted from the [Timeseries clustering](https://github.com/tejaslodaya/timeseries-clustering-vae) repository developed by [Tejas Lodaya](https://tejaslodaya.com).

## VAME in a Nutshell

VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/)-based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution, we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every step of the input time series.
The workflow of VAME consists of 5 steps and we explain them in detail [here](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow)

## Installation

To get started we recommend using [Anaconda](https://www.anaconda.com/distribution/) with Python 3.6 or higher. Here, you can create a [virtual enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to store all the dependencies necessary for VAME. You can also use the VAME.yaml file supplied here, by simply opening the terminal, running git clone https://github.com/LINCellularNeuroscience/VAME.git, then typ cd VAME then run: conda env create -f VAME.yaml).

* Go to the locally cloned VAME directory and run python setup.py install in order to install VAME in your active conda environment.
* Install the current stable Pytorch release using the OS-dependent instructions from the [Pytorch website](https://pytorch.org/get-started/locally/). Currently, VAME is tested on PyTorch 1.5. (Note, if you use the conda file we supply, PyTorch is already installed and you don't need to do this step.)

## Getting Started
First, you should make sure that you have a GPU powerful enough to train deep learning networks. In our original 2022 paper, we were using a single Nvidia GTX 1080 Ti GPU to train our network. A hardware guide can be found [here](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/). VAME can also be trained in Google Colab or on a HPC cluster. Once you have your computing setup ready, begin using VAME by following the [workflow guide](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow).

If you want to follow an example first, you can download video-1 [video-1](https://drive.google.com/file/d/1w6OW9cN_-S30B7rOANvSaR9c3O5KeF0c/view?usp=sharing) and find the .csv file in our [example folder](https://github.com/LINCellularNeuroscience/VAME/tree/master/examples)

Once you are up and running, you can try VAME out on a set of mouse behavioral videos and .csv files publicly available at https://github.com/VAMETools/Miller2024_Data. (coming soon in April 2024)

## References
Original VAME publication: [Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v2) <br/>
Kingma & Welling: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) <br/>
Pereira & Silveira: [Learning Representations from Healthcare Time Series Data for Unsupervised Anomaly Detection](https://www.joao-pereira.pt/publications/accepted_version_BigComp19.pdf)

## License: GPLv3
See the [LICENSE file](https://github.com/LINCellularNeuroscience/VAME/blob/master/LICENSE) for the full statement.

## Code Reference (DOI)
