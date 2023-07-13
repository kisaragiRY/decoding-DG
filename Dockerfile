FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKDIR=/work 
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# necessary packages
RUN apt-get update
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

# download python 3.8.5
RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tar.xz 
RUN tar xJf Python-3.8.5.tar.xz &&\
    cd Python-3.8.5 &&\
    ./configure &&\
    make install &&\
    cd ../ &&\
    rm -rf Python*

# poetry
COPY poetry.lock pyproject.toml ./
ENV POETRY_HOME=/usr/local/poetry \
    POETRY_VERSION=1.4.0
RUN /usr/local/bin/python3.8 -m pip install --upgrade pip \
    && curl -sSL https://install.python-poetry.org | python3 - 
RUN echo 'export PATH="$POETRY_HOME/bin:$PATH"' >> $HOME/.bashrc && \
    $POETRY_HOME/bin/poetry config virtualenvs.create false && \
    $POETRY_HOME/bin/poetry install --no-root

# Create the user with the same uid and gid as the host server (tmp solution)
ARG USERNAME=developer
ARG USER_UID=202003
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME &&\
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME &&\
    # add the $USERNAME to root group
    usermod -aG root $USERNAME &&\
    apt-get update &&\
    apt-get install -y apt-utils sudo &&\
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME &&\
    chmod 0440 /etc/sudoers.d/$USERNAME 
    
USER $USERNAME
ENV HOME=/home/$USERNAME

# setup poetry and python path for the new user
RUN echo 'export PYTHONPATH="$WORKDIR:$WORKDIR/modules:$PYTHONPATH"' >> $HOME/.bashrc && \
    echo 'export PATH="$POETRY_HOME/bin:$PATH"' >> $HOME/.bashrc

# for jupyter env
RUN ipython profile create &&\
    echo "c.InteractiveShellApp.exec_lines = ['import sys; sys.path+=[\"${WORKDIR}\",\"${WORKDIR}/modules\"]']" \
    >> $HOME/.ipython/profile_default/ipython_config.py