# cmrxrecon-kspaceekings BASELINE

## Configuration
```bash
$ conda env create -f environment.yml
$ conda activate cmrxrecon
```

## Run Trainning
```bash
$ export CMR_DATA_PATH=</path/to/CMR/data/CMRxRecon>
$ python run_net.py --root_dir $CMR_DATA_PATH #>> --mode train <<(Optional and useless)
```

## Run Inference
```bash
$ python run_net.py --mode eval --root_dir $CMR_DATA_PATH
```