# LAN_pipeline_minimal
Minimal version of the LAN pipeline for internal purposes

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