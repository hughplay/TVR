# TVR - Transformation driven Visual Reasoning

**| [Homepage](https://hongxin2019.github.io/TVR/) | [Paper](https://arxiv.org/pdf/2011.13160) | [Dataset](https://hongxin2019.github.io/TVR/dataset) | [Explore Data](https://hongxin2019.github.io/TVR/explore) |**


**Motivation:** Most existing visual reasoning tasks, such as CLEVR in VQA, are solely deﬁned to test how well the machine understands the concepts and relations within static settings, like one image. We argue that this kind of **state driven visual reasoning** approach has limitations in reﬂecting whether the machine has the ability to infer the dynamics between different states, which has been shown as important as state-level reasoning for human cognition in Piaget’s theory.

**Task:** To tackle aforementioned problem, we propose a novel **transformation driven visual reasoning** task. Given both the initial and final states, the target is to infer the corresponding single-step or multi-step transformation.

<p align="center">
    <img src="imgs/web.svg" width="500">
</p>


## Dataset Download

The instruction of [downloading our TRANCE dataset](https://hongxin2019.github.io/TVR/dataset) can be found in our homepage.

## Dataset Generation

If you are instrested in generating samples by yourselves or constructing a new dataset based on TRANCE. You can read the [instructions and tips](data/gen_src) for data generation.

## Model Training and Evaluation

The code for model training and evaluation is located in the `src` folder.

### Environments

You can create an isloated python environment by running:

```
cd src
conda create -n tvr python=3.7
pip install -r requirements.txt
```

### Preprocessing the data

You should first download TRANCE from Kaggle, and then preprocess the data with the following command:

```
python core.py preprocess </path/to/trance>
```

### Training & testing

We provide experimental configurations in `src/config`.
If you want to try you own models or modify some parameters, please check these configurations.

Currently, we only support single gpu training and testing.

``` bash
# training
python core.py train config/event.ConcatResNet.yaml

# training and testing on cuda:0
python core.py train config/event.ConcatResNet.yaml --device 'cuda:0' --test

# test only
python core.py train config/event.ConcatResNet.yaml
```

Notice: We fixed a bug in TRANCE, therefore, the performance on Event and View is slightly higher (0.03~0.06 on Acc) than the results reported in our CVPR paper.

## Citing TVR

If you find TVR useful for your research then please cite:

```
@inproceedings{hong2021tvr,
    author={Hong, Xin and Lan, Yanyan and Pang, Liang and Guo, Jiafeng and Cheng, Xueqi},
    title={Transformation Driven Visual Reasoning},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```


## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>

Notice: Some materials are directly inherited from [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen) which are licensed under BSD License. More details can be found in [this document](data/gen_src/resource/README.md).
