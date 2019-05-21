# muir-release


## Installation
```
cd muir-release
mkdir results
pip install -r requirements.txt
```

## Datasets

The code assumes datasets are downloaded into `~/hyperdatasets/<dataset_name>`, e.g., `~/hyperdatasets/cifar` and `~/hyperdatasets/wikitext2`.

Dataset files for the synthetic dataset are included directly in `muir-release/datasets/synthetic`.

Dataset files for Cifar can be downloaded directly with PyTorch.

Dataset files for WikiText-2 can be downloaded from https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/.

## Running Optimization

```
cd muir-release/pytorch/optimize
python optimize.py --experiment_name <exp_name> --config <config_file> --device <device_id>
```

`experiment_name` is the name of the experiment and can be anything. Experiment launch time information will be appended to this name.

`config` is a path to the config file. For example configs, see `muir-release/pytorch/configs`.

`device` is the name of the device for running torch, e.g., `cpu`, `cuda:0`, `cuda:1`, ...

Results for the experiment will be saved to a directory with the experiments name in `muir-release/results`.

## Implementing new Experiments

To use a new architecture, a model class must be implemented that replaces layers with hyperlayers (see `muir-release/pytorch/models/` for examples).

Currently, layers supported for reparameterization by hypermodules are fully-connected, conv2d, conv1d, and LSTM (see `muir-release/pytorch/layers/`). These can be extended to more layer types by following the examples there.

To use a new dataset, it must be implemented to follow the interface of the examples in `muir-release/pytorch/datasets/`.

