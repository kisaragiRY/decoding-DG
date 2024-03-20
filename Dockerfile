FROM python:3.10-slim

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKDIR=/work 
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# necessary packages
RUN apt-get update
RUN apt-get install -y \
    wget curl git vim gcc python3-dev \
    # for scipy
    libxml2-dev libxmlsec1-dev

# poetry
COPY poetry.lock pyproject.toml ./
ENV POETRY_HOME=/usr/local/poetry \
    POETRY_VERSION=1.4.0
RUN /usr/local/bin/python3 -m pip install --upgrade pip \
    && curl -sSL https://install.python-poetry.org | python3 - 
RUN echo 'export PATH="$POETRY_HOME/bin:$PATH"' >> $HOME/.bashrc && \
    $POETRY_HOME/bin/poetry config virtualenvs.create false && \
    $POETRY_HOME/bin/poetry install --no-root


# setup poetry and python path for the new user
# RUN echo 'export PYTHONPATH="$WORKDIR:$WORKDIR/modules:$PYTHONPATH"' >> $HOME/.bashrc && \
#     echo 'export PATH="$POETRY_HOME/bin:$PATH"' >> $HOME/.bashrc

# for jupyter env
RUN ipython profile create &&\
    echo "c.InteractiveShellApp.exec_lines = ['import sys; sys.path+=[\"${WORKDIR}\",\"${WORKDIR}/modules\"]']" \
    >> $HOME/.ipython/profile_default/ipython_config.py