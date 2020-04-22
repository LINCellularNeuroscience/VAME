![VAME](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/VAME_Logo.png)
![workflow](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/workflow.png)

# VAME in a Nutshell
VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/) based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every time step. 

The workflow of VAME consists of 5 steps and we explain them in detail [here](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow).

## Getting Started
To get started we recommend using [Anaconda](https://www.anaconda.com/distribution/) with Python 3.6 or higher. 
Here, you can create a [virtual enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to store all the dependencies necessary for VAME.

First, you should make sure that you have a GPU powerful enough to train deep learning networks. In our paper, we were using a single Nvidia GTX 1080 Ti to train our network. A hardware guide can be found [here](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/). Once you have your hardware ready, follow the [PyTorch](https://pytorch.org/get-started/locally/) installation guide to set up PyTorch for your virtual anaconda enviroment. 
Next, you can install the setup.py file in your enviroment. 

## News
* April 2020: Preprint coming soon

### Authors and Code Contributors
VAME was developed by Kevin Luxem and Pavol Bauer

### References
If you use this code or data please cite

### License: GPLv3
See the [LICENSE file](../master/LICENSE) for the full statement.
