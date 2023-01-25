# TVR - Transformation driven Visual Reasoning

<br>

Official repository for ["Transformation driven Visual Reasoning"](https://github.com/hughplay/TVR).

<!-- ![A fancy image here](docs/_static/imgs/logo.svg) -->
<img src="docs/_static/imgs/web.svg" width="500">

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


<br>

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


<br>

## Environment Setup

We use docker to manage the environment. You need to build the docker image first and then enter the container to run the code.

**0. Basic Setup**

The host machine should have installed following packages (need sudo previlege to install):
- [Docker](https://docs.docker.com/engine/install)
- [Nvidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

And we also need to install [docker-compose](https://docs.docker.com/compose/install/) to manage the docker containers. Simply run the following command to install it:

```bash
pip install docker-compose
```

**1. Start the docker container**

```
python docker.py startd
```

Please follow the prompts to set the variables such as `PROJECT`, `DATA_ROOT`, `LOG_ROOT`. After that, a `.env` file will be generated in the root directory. You can also modify the variables in the `.env` file directly.

`DATA_ROOT` is mapped to `/data` in the container. `LOG_ROOT` is mapped to `/log` in the container.

**Tips:** when you first start the container, it will take some time to build the image. After that, it will be much faster. If you want to rebuild the image, you can run:

```
python docker.py startd --build
```


**2. Enter the container**

``` sh
python docker.py
# or
python docker.py enter
```

<br>

## Data Preparation

**1. Download TRANCE dataset**

Follow the [steps in this page](https://hongxin2019.github.io/TVR/dataset) to download TRNACE from Kaggle. After that, decompress the package under `DATA_ROOT` (specified in `.env` file). The final dataset location should be `DATA_ROOT/trance` in the host machine and it will be mapped to `/data/trance` in the container.

**2. Preprocess the data**

Preprocess the data with the following command:

```
python scripts/preprocess.py /data/trance
```

This will merge the raw images files and meta data into a single hdf5 file. After that, the directory should include the following files:


```
trance
├── data.h5
├── properties.json
└── values.json
```


<br>

## Training & testing

After entering the container, you can run the following command to train a model:

``` bash
python train.py experiment=event_cnn_concat logging.wandb.tags="[event, base]"
```

Or, you can training multiple models with provided GPUs:

``` bash
python scripts/batch_train.py scripts/training/train_models.sh --gpus 0,1,2,3
```

Please refer to the scripts under [`scripts/training`](scripts/training) for full training commands.

**Notice:** We fixed a bug in TRANCE, therefore, the performance on Event and View is slightly higher (0.03~0.06 on Acc) than the results reported in our CVPR paper.


<br>


## Demo

We provide a demo to explore the dataset and testing predictions of trained models.

![](docs/_static/imgs/demo.png)

**1. Launch the api server**

Enter the project container and run the api server:

``` bash
python docker.py
uvicorn src.demo.api_server.main:app --host 0.0.0.0 --port 8000 --reload
```

**Tips 1:** you can check the api docs by visiting `http://<host_ip>:8000/docs` in your browser. The `host_ip` is the ip address of the host machine.

**Tips 2:** the default port is 8000. If you use another port, you need also to modify the port specified in [`src/demo/ui/src/js/api.js`](src/demo/ui/src/js/api.js).

**2. Launch the web (UI) server**

We need another docker container to launch the ui. Run the command in another terminal window in the host machine (recommend tmux):

``` sh
python docker.py start --service demo
```

When you first start the container, besides the image building, it will also take some time to install the npm packages. After that, it will be much faster.

<br>

## Data Generation

We provide the code to generate the dataset.

**1. Build the docker image**

```
python docker.py prepare --service blender --build
```

**2. Enter the container**

```
python docker.py --service blender
```

**3. Generate the dataset**

``` sh
# with CPU
blender --background --python render.py -- --config configs/standard.yaml --gpu false --render_tile_size 16

# with GPU
CUDA_VISIBLE_DEVICES=0 blender --background --python render.py -- --config configs/standard.yaml --gpu true --n_sample 1
```

The speed of rendering can be affected by:
- GPU or CPU. Gererally, GPU is more faster than CPU, unless your CPU has many cores.
- `render_tile_size`. CPU prefers small tile size, while GPU prefers large tile size.
- Balanced sampling. It has noting to do with blender rendering. However, sampling scene graph for rendering can also be time consuming.

<br>

## LICENSE

The code is licensed under the [MIT license](./LICENSE) and the TRANCE dataset is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

Notice: Some materials are directly inherited from [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen) which are licensed under BSD License. More details can be found in [this document](trance/resource/README.md).

<br>

*This is a project based on [DeepCodebase](https://github.com/hughplay/DeepCodebase) template.*
