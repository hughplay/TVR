# reference: https://github.com/Vogete/blender-cuda-docker/blob/master/blender2.79b/cuda10.0/ubuntu18.04/Dockerfile
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
MAINTAINER Mr.Blue

ENV BLENDER_VERSION 2.79
ENV BLENDER_BZ2_URL https://download.blender.org/release/Blender${BLENDER_VERSION}/blender-${BLENDER_VERSION}-linux-glibc219-x86_64.tar.bz2

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    curl \
    wget \
    bzip2 \
    libfreetype6 \
    libgl1-mesa-dev \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV BLENDER_DIR /usr/local/blender

RUN mkdir ${BLENDER_DIR} && \
	wget --quiet ${BLENDER_BZ2_URL} -O blender.tar.bz2 && \
	tar -jxvf blender.tar.bz2 -C ${BLENDER_DIR} --strip-components=1 && \
	rm blender.tar.bz2

ENV BLENDER_PYTHON ${BLENDER_DIR}/2.79/python/bin/python3.5m

RUN ${BLENDER_PYTHON} -m ensurepip
RUN ${BLENDER_PYTHON} -m \
    pip install --no-cache-dir -U pip \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -i http://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt /tmp/requirements.txt
RUN rm -rf ${BLENDER_DIR}/2.79/python/lib/python3.5/site-packages/numpy*
RUN /usr/local/blender/2.79/python/bin/python3.5m -m \
    pip install --no-cache-dir \
    -r /tmp/requirements.txt \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -i http://pypi.tuna.tsinghua.edu.cn/simple \
    && rm -f /tmp/requirements.txt

RUN ln -s ${BLENDER_DIR}/blender /usr/local/bin/blender

ARG USER_ID=1011
ARG GROUP_ID=1011
ARG USER_NAME=docker
RUN if [ $USER_NAME != "root" ] ; \
    then addgroup --gid ${GROUP_ID} ${USER_NAME} \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ; fi

# Switch to the created user
USER ${USER_NAME}

# Set working directory to /workspace
WORKDIR "/workspace"
