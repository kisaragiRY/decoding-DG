# Dentate Gyrus Encoding Decoding
Author: Kisaragi

ROCKET (RandOm Convolutional KErnel Transform) implementation on sparse spiking data from mice's Dentate Gyrus(DG).

## Project Setup
1. add `.env` file to the root folder

    ```txt
    PYTHONPATH=/work
    RAW_DATA_PATH=[your raw data path]
    ```
2. docker env setup
    ```bash
    cd [the repo location]
    docker-compose build
    docker-compose up -d
    ```
3. attach the container to vscode editor
4. install plugin in vscode
    - Jupyter
    - Python
5. prepare data: run `/notebooks/1-data-preparation/xlsx2csv.py`
6. run training script: `/notebooks/3-decoding-model/1-varied-num-of-samples/train.py`
7. run the shuffle training script: `/notebooks/3-decoding-model/1-varied-num-of-samples/shuffle-train.py`
    

## Repo structure
- `/modules`:
    - includes classes and functions for datasets, information metrics, etc.
- `/notebooks`:
    - includes data exploration (behavioural analysis and neural analysis) and decoding model implementation.
- `Dockerfile`: 
    - project envoironment.
- `poetry.lock` & `pyptoject.toml`:
    - project Python dependencies.



