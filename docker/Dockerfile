# ==================================================
# Based on deepbase/project:codebase
# You can find the Dockerfile from:
# https://github.com/deepcodebase/docker/blob/master/project/codebase/Dockerfile
#
# Alternatively, you can build the image from scratch by using Dockerfile.full
# ==================================================

FROM deepbase/project:codebase


# ==================================================
# Install python packages
# ==================================================

# Setup TUNA mirror (optional)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
COPY misc/.condarc /root/.condarc

# By default, install packages from `requirements.txt` with pip.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt


# Another way is installing packages from a `env.yaml` with conda.
# COPY env.yaml /tmp/env.yaml
# RUN conda env create -f /tmp/env.yaml && rm -f /tmp/env.yaml


# ==================================================
# Post-installation steps
#
# Create a user that has the same UID and GID as the host user. This will
# prevent many privileges issues.
# ==================================================

# COPY .localrc
COPY misc/.localrc /root
RUN echo ". ~/.localrc" >> ~/.zshrc


# Add a user with the same UID and GID as the host user, to prevent privilege issues.
ARG USER_ID=1011
ARG GROUP_ID=1011
ARG USER_NAME=docker
RUN if [ $USER_NAME != "root" ] ; \
    then addgroup --gid ${GROUP_ID} ${USER_NAME} \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ; fi


# Copy installed configuration files from root to user
COPY misc/init_workspace /usr/local/bin
RUN /usr/local/bin/init_workspace --user ${USER_NAME} --home /home/${USER_NAME}
RUN cp -r /home/${USER_NAME} /${USER_NAME}_home_bak


# Switch to the created user
USER ${USER_NAME}


# Set working directory to /workspace
WORKDIR "/workspace"
