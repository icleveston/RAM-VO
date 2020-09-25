# Recurrent-Models-of-Visual-Attention-TF-2.0
This repository contains the a modified Recurrent Attention Model which was described in the Paper Recurrent Models of Visual Attention. 

- `bayesian_opt/` contains scripts for bayesian hyperparameter tuning on every dataset
- `data/` contains scripts for loading data e.g. bach dataset loader, mnist, ...
- `example/` contains notebooks on how to use all modules
- `model/` contains implementation of the whole model
    - `ram.py` contains the implementation of the Recurrent Attention Model
    - `layers.py` contains the implementation of the convolution layer (change this to try out other convolutions)
- `visualizations/` contain scripts for visualizing the model and data
- `./` contains jupyter notebooks about how to use the dataloader, how to use the visualization scripts and how to train the model

## Requirements (everything will be installed with the requirement.txt)
- [ray](http://ray.readthedocs.io)
- [bayesian opt](https://github.com/fmfn/BayesianOptimization)
- [tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
- numpy, matplotlib, scikit-learn

## Getting Started
In order to run this code is it **recommended** to use the docker container of tensorflow 2.0.0a because it includes all the needed drives etc. You need to install `tf-nightly` though because tensorflow 2.0.0a does not support tensorflow probabilty.

```bash
nvidia-docker run -it --rm -p 8888:8888 tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter bash
pip install -r requirements.txt
git clone git@git.tools.f4.htw-berlin.de:smi/recurrent-visual-attention-model.git

cd recurrent-visual-attention-model
jupyter notebook
```
**Note:** If you do not have a GPU then you can remove the `gpu` tag and replace `nvidia-docker` with `docker`


## Modifications
- instead of translating and adding clutter while runtime, data loaders were created where this process is done only once. 
  - it is possible to test that the RAM is can archive a good performance with limited data
  - but you can also create the dataset after each epoch to simulate the creation via runtime
- instead of Dense layers/Fully Conneted layers, Convolution layers were used
- in addition to the baseline model, batch norm was added to reduce variance
- instead of random search, Bayessian Hyperparameter Optimization was used to tune the hyperparameter of the network (std and initial learning rate)



## Results
**Note:** Every model was trained with ADAM optimizer instead of SGD with momentum

| Dataset                            | Model                    | Hyperparameter                        | Epochs | Error |
|------------------------------------|--------------------------|---------------------------------------|--------|-------|
| MNIST                              | 1 8x8 Glimpse, 7 steps   | 0.25 STD, 0.001 LR, 1.0 max gradient  | 200    | 1.9%  |
| Transalted MNIST                   | 3 12x12 Glimpse, 8 steps | 0.05 STD, 0.0001 LR, 5.0 max gradient | 1000   | 2.83% |
| Cluttered Translated MNIST 60x60   | 3 12x12 Glimpse, 8 steps | TODO                                  | TODO   | TODO  |
| Cluttered Transalted MNIST 100x100 | 4 12x12 Glimpse, 8 steps | TODO                                  | TODO   | TODO  |

## Some Words
The paper Recurrent Models of Visual Attention is 5 years and received since then a lot of modification. I think the REINFORCE algorithm still a interesting "cheat" or "trick" to optimize for non differentiable variables which is why I tried to implement and understand it. This implementation also has a very object oriented style, thus every class/module can be swapped out easily.
