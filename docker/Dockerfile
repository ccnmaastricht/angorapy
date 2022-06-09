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
RUN apt-get update

WORKDIR /usr/local/cuda-11.2/lib64
RUN ln -s libcusolver.so.11 libcusolver.so.10
WORKDIR /

# install packages
RUN apt-get update && apt-get install -y \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# INSTALL MPI
RUN apt-get update
RUN apt-get install -y file g++ gcc make gdb strace ca-certificates --no-install-recommends
RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz
RUN tar xf mpich-3.1.4.tar.gz
WORKDIR mpich-3.1.4
RUN ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr
RUN make -j$(nproc)
RUN make install
RUN ldconfig
WORKDIR /

# INSTALL PYTHON
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.8-dev python3.8 python3-pip

# ENVIRONMENT VARIABLES
ENV LD_LIBRARY_PATH="/usr/lib/nvidia:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH"

# matplotlib
RUN mkdir /mplconfs
ENV MPLCONFIGDIR="/mplconfs"

# SETUP ENV
ENV VIRTUAL_ENV=/venv
RUN virtualenv --python=python3.8 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# INSTALL TOOL
RUN mkdir /ang_tmp
COPY . /ang_tmp
RUN pip install /ang_tmp
RUN pip uninstall mujoco-py
RUN rm -rf /ang_tmp

WORKDIR /
RUN ls -la
COPY angorapy/train.py train.py

# FINAL
CMD [ "/bin/bash" ]

