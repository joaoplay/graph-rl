FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

MAINTAINER João Brás Simões <joaosbraz@gmail.com>

# Ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PYTHON_VERSION 3.9

ENV DEBIAN_FRONTEND=noninteractive

ENV FORCE_CUDA=1

ARG USE_CUDA=0
ENV USE_CUDA=${USE_CUDA}

WORKDIR /deps

RUN apt-get update && apt-get install -y  build-essential zlib1g-dev \
libncurses5-dev libgdbm-dev libnss3-dev \
libssl-dev libreadline-dev libffi-dev curl software-properties-common cmake wget git

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3.9-distutils

RUN apt-get install -y intel-mkl

RUN cd /usr/local/bin \
    && ln -s /usr/bin/python3.9 python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3

RUN curl -sSL https://install.python-poetry.org | python3 -

ADD ./poetry.lock /deps/
ADD ./pyproject.toml /deps/

RUN /root/.local/bin/poetry config virtualenvs.in-project true

RUN /root/.local/bin/poetry install --no-interaction --no-ansi

ENV PATH="/code/.venv/bin:$PATH"

RUN cd /usr/lib \
    && git clone https://github.com/joaoplay/pytorch_structure2vec.git \
    && cd pytorch_structure2vec \
    && cd s2v_lib \
    && make -j4

VOLUME ["/code"]

WORKDIR /code