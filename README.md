![VAME](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/VAME_Logo-1.png)
![workflow](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/workflow.png)

# VAME in a Nutshell
VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/) based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every step of the input time series.

![behavior](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/behavior_structure_crop.gif)

The workflow of VAME consists of 5 steps and we explain them in detail [here](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow).

## Installation
To get started we recommend using [Anaconda](https://www.anaconda.com/distribution/) with Python 3.6 or higher. 
Here, you can create a [virtual enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to store all the dependencies necessary for VAME. (you can also use the VAME.yaml file supplied here, byt simply openning the terminal, running `git clone https://github.com/LINCellularNeuroscience/VAME.git`, then type `cd VAME` then run: `conda env create -f VAME.yaml`).

* Go to the locally cloned VAME directory and run `python setup.py install` in order to install VAME in your active conda environment.
* Install the current stable Pytorch release using the OS-dependent instructions from the [Pytorch website](https://pytorch.org/get-started/locally/). Currently, VAME is tested on PyTorch 1.5. (Note, if you use the conda file we supply, PyTorch is already installed and you don't need to do this step.)

## Getting Started
First, you should make sure that you have a GPU powerful enough to train deep learning networks. In our paper, we were using a single Nvidia GTX 1080 Ti to train our network. A hardware guide can be found [here](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/). Once you have your hardware ready, try VAME following the [workflow guide](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow).

## News
* March 2021: We are happy to release VAME 1.0 with a bunch of improvements and new features! These include the community analysis script, a model allowing generation of unseen datapoints, new visualization functions, as well as the much requested function to generate GIF sequences containing UMAP embeddings and trajectories together with the video of the behaving animal. Big thanks also to [@MMathisLab](https://github.com/MMathisLab) for contributing to the OS compatibility and usability of our code.
* November 2020: We uploaded an egocentric alignment [script](https://github.com/LINCellularNeuroscience/VAME/blob/master/examples/align_demo.py) to allow more researcher to use VAME
* October 2020: We updated our manuscript on [Biorxiv](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v2)
* May 2020: Our preprint "Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion" is out! [Read it on Biorxiv!](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v1)

### Authors and Code Contributors
VAME was developed by Kevin Luxem and Pavol Bauer.

The development of VAME is heavily inspired by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/).
As such, the VAME project management codebase has been adapted from the DeepLabCut codebase.
The DeepLabCut 2.0 toolbox is © A. & M.W. Mathis Labs [deeplabcut.org](http:\\deeplabcut.org), released under LGPL v3.0.

### References
VAME preprint: [Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v1)

### License: GPLv3
See the [LICENSE file](../master/LICENSE) for the full statement.
