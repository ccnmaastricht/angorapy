ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.2.2-1
ARG LIBNVINFER_MAJOR_VERSION=7

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        libcudnn8=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
# NOTE: libnvinfer uses cuda11.1 versions
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub && \
        echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"  > /etc/apt/sources.list.d/tensorRT.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0 \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

RUN pip install --upgrade pip

# INSTALL TOOL
RUN mkdir /ang_tmp
COPY . /ang_tmp

RUN pip install /ang_tmp \
    && pip install mujoco \
    && pip install imageio

# fix missing stls problem
RUN cp -r /ang_tmp/angorapy/environments/assets/stls/ venv/lib/python3.8/site-packages/angorapy/environments/assets/

RUN rm -rf /ang_tmp

RUN pip install tensorflow_datasets
RUN pip install gym[box2d]

WORKDIR /
COPY angorapy/train.py train.py
COPY angorapy/pretrain.py pretrain.py

# CLEANUP
RUN pip cache purge

# FINAL
CMD [ "/bin/bash" ]

