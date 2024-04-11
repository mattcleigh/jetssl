<div align="center">

# JetSSL

[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.1-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB_0.16-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
</div>

This project is generated from the RODEM template for training deep learning models using PyTorch, Lightning, Hydra, and WandB. It is loosely based on the PyTorch Lightning Hydra template by ashleve.

## Submodules

This project relies on a custom submodule called `mltools` stored [here](https://gitlab.cern.ch/mleigh/mltools/-/tree/master) on CERN GitLab.
This is a collection of useful functions, layers and networks for deep learning developed by the RODEM group at UNIGE.

If you didn't clone the project with the `--recursive` flag you can pull the submodule using:

```
git submodule update --init --recursive
```

## Configuration

All job configuration is handled by hydra and omegaconf and are stored in the `configs` directory.
The main configuration file that composes training is `train.yaml`.
It sets the seed for reproducibility, the project and network names, the checkpoint path for resuming training, precision and compile options for PyTorch, and tags for the loggers.
This file composes the training config using yaml files for the `trainer`, `model`, `datamodule`, `loggers`, `paths`, `callbacks`.
The `experiment` folder is used to overwite any of the config values before composition.
Ideally trainings should always be run using `python train.py experiment=...`

## Usage

To run this project, follow these steps:

1. Setup your environment:
* Pull the docker image  `gitlab-registry.cern.ch/mleigh/jetssl:latest`
* OR or install the `requirements.txt` in a virtual environment of your choice
    * Requires `python > 3.10`

2. Setup Data
* This project is designed to work with two datasets which must be downloaded and converted into the correct format.
    1. JetClass
        * Download from `https://zenodo.org/records/6619768`
        * Convert using `scripts/make_jetclass.py`
    2. Secondary Vertex Finding in Jets Dataset (Shlomi)
        * Download from `https://zenodo.org/records/4044628`
        * Convert using `scripts/make_shlomi.py`
* All my paths are hardcoded at the moment so you will have to change them
    * This is something that will be fixed in a later update
    * You can find all the hardcoded paths by
        * `grep -r "/srv/"`

3. Train
* Run the training script with the desired configuration:
```
python scripts/train.py experiment=train_mpm.yaml
```

## Docker and Gitlab

This project is setup to use the CERN GitLab CI/CD to automatically build a Docker image based on the `docker/Dockerfile` and `requirements.txt`.
It will also run the pre-commit as part of the pipeline.
To edit this behaviour change `.gitlab-ci`

## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you have any improvements or suggestions.
Please use the provided `pre-commit` before making merge requests!

## License

This project is licensed under the MIT License. See the [LICENSE](https://gitlab.cern.ch/rodem/projects/projecttemplate/blob/main/LICENSE) file for details.
