FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home\
    WORKDIR=/work 
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt-get update

# necessary packages
RUN apt-get install -y \
    wget curl git gcc make vim\
    build-essential \
    libsm-dev \
    libxrender-dev \
    libxext-dev \
    libgl1-mesa-dev \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python-dev \
    python3-venv \
    libssl-dev \
    zlib1g-dev \
    xz-utils \
    llvm \
    libncurses5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    libreadline-dev \
    libsqlite3-dev \
    python3-tk \
    libbz2-dev

# download python
RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tar.xz \
    && tar xJf Python-3.8.5.tar.xz \
    && cd Python-3.8.5 \
    && ./configure \
    && make install \
    && cd ../ \
    && rm -rf Python*
RUN echo 'export PYTHONPATH="$WORKDIR:$PYTHONPATH"' >> $HOME/.bashrc 


# poetry
COPY poetry.lock pyproject.toml ./
ENV POETRY_HOME=/usr/local/poetry \
    POETRY_VERSION=1.2.0
RUN /usr/local/bin/python3.8 -m pip install --upgrade pip \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 - \
    && echo 'export PATH="$POETRY_HOME/bin:$PATH"' >> $HOME/.bashrc \
    && $POETRY_HOME/bin/poetry config virtualenvs.create false \
    && $POETRY_HOME/bin/poetry install --no-root

# add user
USER zhang-r