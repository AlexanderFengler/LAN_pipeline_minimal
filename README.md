# LAN_pipeline_minimal
Minimal version of the LAN pipeline for internal purposes

## Installation Instructions

To install an environment that will be sufficient for running the whole pipeline, you can use the following steps:

(This assumes you have the `mamba` package manager installed in the base environment of your `conda` package manager,
if you don't have `mamba` installed create the new environment with conda instead)

```
mamba create -n lan_pipe python=3.10
conda activate lan_pipe
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lanfactory
pip install ssm-simulators
pip install jupyter
```

If you want to make your new environment available in e.g. Visual Studio Code, run also,

```
python -m ipykernel install --user --name=lan_pipe
```

### Oscar specifics

In your oscar home directory, create a `.modules` file (if you don't already have it, in which case just add the lines below to it), and enter the following content,

```
module load git/2.29.2
module load gcc/10.2
module load cuda/11.8.0
module load cudnn/8.6.0
```

## Usage

The pipeline works as a two-step process.

1. Data generation

2. Network training

Check the `config_constructor.ipynb` notebook to get guidance on how to create config-files for both *data generation*, as well as *network training* runs.

Once you have these configs for your current run of interest (you can reuse training data generated for as many *network training* runs as you like),
you can use the:

1. `bash_data_generation.sh` script to run data generation on the Oscar (specialize the script to use your login and base folders, the rest of the settable parameters pick out the correct data generation config file)

2. `bash_network_training.sh` script to run network training on Oscar (this allows you to pick gpu / cpu backend as well as choice amongst jax / pytorch)

The `local_network_training.sh` script helps with training networks on your local machine instead.

The scripts will automatically generate folders that store the config files, training data and network meta-data as well as trained networks.

You should see a fairly straightforwardly navigable `data` folder appear in the **project folder** you specify in the config and bash scripts.