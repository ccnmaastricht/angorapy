ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=11.2

FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-cudnn8-runtime-ubuntu${UBUNTU_VERSION} as base

ARG DEBIAN_FRONTEND=noninteractive

# key fix
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*
RUN apt-key del 7fa2af80

COPY docker/cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

WORKDIR /usr/local/cuda-11.2/lib64
RUN ln -s libcusolver.so.11 libcusolver.so.10
WORKDIR /

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    swig \
    wget \
    virtualenv \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libc6-dev \
    libosmesa6-dev \
    patchelf \
    software-properties-common \
    gfortran \
    && apt-get install --no-install-recommends -y file g++ gcc make gdb strace ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get --no-install-recommends install -y python3.8-dev python3.8 python3-pip \
    && apt-get clean
#    && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y libopenexr-dev
RUN apt-get install -y openexr
RUN apt-get install -y zlib1g-dev

# INSTALL MPI
RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz
RUN tar xf mpich-3.1.4.tar.gz
WORKDIR mpich-3.1.4
RUN ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr
RUN make -j$(nproc)
RUN make install
RUN ldconfig
WORKDIR /

# ENVIRONMENT VARIABLES
ENV LD_LIBRARY_PATH="/usr/lib/nvidia:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH"

# matplotlib
ENV MPLCONFIGDIR=/var/cache/matplotlib

# SETUP ENV
ENV VIRTUAL_ENV=/venv
RUN virtualenv --python=python3.8 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# INSTALL TOOL
RUN mkdir /ang_tmp
COPY . /ang_tmp

RUN pip install /ang_tmp --default-timeout=100 \
    && rm -rf /ang_tmp

RUN pip install dm_control==1.0.12
RUN pip install tensorflow_datasets
RUN pip uninstall -y mujoco-py
RUN pip install mujoco_utils

WORKDIR /

COPY angorapy/train.py train.py
COPY angorapy/pretrain.py pretrain.py
COPY angorapy/analysis/predictability.py predictability.py

# CLEANUP
RUN pip cache purge

# FINAL
CMD [ "/bin/bash" ]
