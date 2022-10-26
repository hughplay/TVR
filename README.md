# TVR - Transformation driven Visual Reasoning

Official repository for ["Transformation driven Visual Reasoning"](https://github.com/hughplay/TVR).

<!-- ![A fancy image here](docs/_static/imgs/logo.svg) -->
<img src="imgs/web.svg" width="500">

**Figure:** *Given the initial state and the final state, the target is to infer the intermediate transformation.*

> **Transformation driven Visual Reasoning** <br>
> Xin Hong, Yanyan Lan, Liang Pang, Jiafeng Guo, Xueqi Cheng <br>
> *Published on 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*

[![](imgs/project.svg)](https://hongxin2019.github.io/TVR/)
[![](https://img.shields.io/badge/-code-green?style=flat-square&logo=github&labelColor=gray)](https://github.com/hughplay/TVR)
[![](https://img.shields.io/badge/TRANCE-dataset-blue?style=flat-square&labelColor=gray)](https://hongxin2019.github.io/TVR/dataset)
[![](https://img.shields.io/badge/TRANCE-explore_dataset-blue?style=flat-square&labelColor=gray)](https://hongxin2019.github.io/TVR/explore)
[![](https://img.shields.io/badge/arXiv-2011.13160-b31b1b?style=flat-square)](https://arxiv.org/pdf/2011.13160.pdf)
[![](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)


## Description

**Motivation:** Most existing visual reasoning tasks, such as CLEVR in VQA, are solely deﬁned to test how well the machine understands the concepts and relations within static settings, like one image. We argue that this kind of **state driven visual reasoning** approach has limitations in reﬂecting whether the machine has the ability to infer the dynamics between different states, which has been shown as important as state-level reasoning for human cognition in Piaget’s theory.

**Task:** To tackle aforementioned problem, we propose a novel **transformation driven visual reasoning** task. Given both the initial and final states, the target is to infer the corresponding single-step or multi-step transformation.

If you find this code useful, please consider to star this repo and cite us:

```
@inproceedings{hongTransformationDrivenVisual2021d,
  title = {Transformation {{Driven Visual Reasoning}}},
  booktitle = {2021 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author = {Hong, Xin and Lan, Yanyan and Pang, Liang and Guo, Jiafeng and Cheng, Xueqi},
  year = {2021},
  pages = {6899--6908}
}
```

## Environment Setup

You can create an isloated python environment by running:

```
cd src
conda create -n tvr python=3.7
pip install -r requirements.txt
```

### Data Preparation

You should first download TRANCE from Kaggle, and then preprocess the data with the following command:

```
python core.py preprocess </path/to/trance>
```

## Training & testing

We provide experimental configurations in `src/config`.
If you want to try you own models or modify some parameters, please check these configurations.

Currently, we only support single gpu training and testing.

``` bash
# training
python core.py train config/event.ConcatResNet.yaml

# training and testing on cuda:0
python core.py train config/event.ConcatResNet.yaml --device 'cuda:0' --test

# test only
python core.py test config/event.ConcatResNet.yaml
```

Notice: We fixed a bug in TRANCE, therefore, the performance on Event and View is slightly higher (0.03~0.06 on Acc) than the results reported in our CVPR paper.

## LICENSE

The code is licensed under the [MIT license](./LICENSE) and the TRANCE dataset is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

Notice: Some materials are directly inherited from [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen) which are licensed under BSD License. More details can be found in [this document](data/gen_src/resource/README.md).