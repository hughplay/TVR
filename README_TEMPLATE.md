# DeepCodebase

Official repository for ["DeepCodebase for Deep Learning"](https://github.com/hughplay/DeepCodebase).

![A fancy image here](docs/_static/imgs/logo.svg)

**Figure:** *DeepCodebase for Deep Learning.
(Read [DEVELOPMENT.md](./DEVELOPMENT.md) to learn more about this template.)*

> **DeepCodebase for Deep Learning** <br>
> Xin Hong <br>
> *Published on Github*

[![](docs/_static/imgs/project.svg)](https://hongxin2019.github.io)
[![](https://img.shields.io/badge/-code-green?style=flat-square&logo=github&labelColor=gray)](https://github.com/hughplay/DeepCodebase)
[![](https://img.shields.io/badge/arXiv-1234.5678-b31b1b?style=flat-square)](https://arxiv.org)
[![](https://img.shields.io/badge/Open_in_Colab-blue?style=flat-square&logo=google-colab&labelColor=gray)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
[![](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![](https://img.shields.io/badge/-Lightning-792ee5?style=flat-square&logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![](docs/_static/imgs/hydra.svg)](https://hydra.cc)
[![](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square&labelColor=gray)](#license)

## News

- [x] **[2022-07-21]** Initial release of the DeepCodebase.

## Description

DeepCodebase is a codebase/template for deep learning researchers, so that do
experiments and releasing projects becomes easier.
**Do right things with suitable tools!**

***This README.md is meant to be the template README of the releasing project.
[Read the Development Guide](./DEVELOPMENT.md) to realize and start to use DeepCodebase.***

If you find this code useful, please consider to star this repo and cite us:

```
@inproceedings{deepcodebase,
  title={DeepCodebase for Deep Learning},
  author={Xin Hong},
  booktitle={Github},
  year={2022}
}
```

## Environment Setup

This project recommends to run experiments with [docker](https://www.docker.com/).
However, we also provide a way to install the experiment environment with
`conda` directly on the host machine.
Check [our introduction about the environment](./DEVELOPMENT.md#docker---prepare-the-environment)
for details.

### Quick Start

The following steps are to build a docker image for this project and run.

**Step 1.** Install docker-compose in your host machine.
```sh
# (set PyPI mirror is optional)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install docker-compose
```

**Step 2.** Build a docker image according to `docker/Dockerfile` and start
a docker container.
```sh
python docker.py prepare --build
```
**When you first run `docker.py`, it will prompt you to set variables such as project
name, data root, etc. These variables are related to the docker container and will
be stored in `.env` file under the root. Read the [DEVELOPMENT.md](./DEVELOPMENT.md)
for more information.*

**Step 3.** Enter the docker container at any time, the environment is ready for experiments.
```sh
python docker.py [enter]
```

### Data Preparation

MNIST will be automatically downloaded to `DATA_ROOT` and prepared by `torch.dataset.MNIST`.


## Training

Commonly used training commands:

```sh
# training a mnist_lenet on GPU 0
python train.py experiment=mnist_lenet devices="[0]"

# training a mnist_lenet on GPU 1
python train.py experiment=mnist_dnn devices="[1]"

# training a mnist_lenet on two gpus, and change the experiment name
python train.py experiment=mnist_lenet devices="[2,3]" name="mnist lenet 2gpus"
```

Read [sections about the configuration](./DEVELOPMENT.md#hydra---configuration)
to learn how to configure your experiments structurally and simply override them
from command line.

## Testing

Commonly used testing commands:

```sh
# test the model, <logdir> has been printed twice (start & end) on training log
python test.py <logdir>
# test the model, with multiple config overrides, e.g.: to test multiple datasets
python test.py <logdir> --update_func test_original test_example
# update wandb, and prefix the metrics
python test.py --update_func test_original test_example --prefix original example --update_wandb
# generate LaTex Tables
python scripts/generate_latex_table.py
```

Read [sections about wandb](./DEVELOPMENT.md#weights--biases-wandb)
to learn how exporting the LaTex table from experimental records works.


## Acknowledgement

Many best practices are learned from [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), thanks to the maintainers of this project.

## License

[MIT License](./LICENSE)

<br>

*The is a [DeepCodebase](https://github.com/hughplay/DeepCodebase) template based project.*
