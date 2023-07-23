# Spectral Sensor Fusion for Prediction of Li and Zr in Rocks: Neural Network and PLS Methods

This repo contains the source code used to run the experiments described in **"Spectral Sensor Fusion for Prediction of Li and Zr in Rocks: Neural Network and PLS Methods"** by Matthew Dirks, David Turner, and David Poole and published in the journal of Chemometrics and Intelligent Laboratory Systems:

- DOI: https://doi.org/10.1016/j.chemolab.2023.104915
- Publisher link: https://www.sciencedirect.com/science/article/pii/S016974392300165X
- My homepage: https://skylogic.ca

**If you use this software for research (or other works), please cite the following paper:**
```bibtex
@article{Dirks2023SensorFusion,
  title = {Spectral sensor fusion for prediction of {Li} and {Zr} in rocks: Neural network and {PLS} methods},
  author = {Matthew Dirks and David Turner and David Poole},
  journal = {Chemometrics and Intelligent Laboratory Systems},
  pages = {104915},
  year = {2023},
  issn = {0169-7439},
  doi = {https://doi.org/10.1016/j.chemolab.2023.104915},
  url = {https://www.sciencedirect.com/science/article/pii/S016974392300165X}
}
```

# Software Requirements

The main requirements are:

- Python 3.9
- Tensorflow 2.6.0

A complete listing of the python package environment (which uses anaconda) is listed in `requirements.txt`.


# Installation

The code in this repo is a Python package. Once you download the repo, link the repo code to your Python environment using:

```bash
cd /path/to/this/git/repo
python setup.py develop
```

Test that it worked by running `import spectra_ml` in Python.

# Dataset

The rock dataset used in the paper cannot be released due to commercial restrictions. The mangoes dataset by Anderson et al [(link to paper)][(https://doi.org/10.1016/j.postharvbio.2020.111202) is included here as an example to demonstrate code functionality.
The dataset is available to download from [Mendeley Data](https://doi.org/10.17632/46htwnp833.2).

In this dataset, visible and near-infrared (Vis-NIR) spectra of mango fruit from four harvest seasons (2015, 2016, 2017, and 2018) were collected. Using spectra from 3 years, the goal is to make predictions of samples in the 4th year.
The spectral bands range 300-1100 nm with approximately 3.3 nm intervals.
Near infrared spectroscopy allows for non-invasive assessment of fruit quality.
In this case, the prediction target is the percent of dry matter (DM) content. 
DM% is an index of total carbohydrates which indicates quality of mango fruit.

You can take a look at the data by calling the load function:
```python
>>> from spectra_ml.components.data_loader import mangoes
>>> d = mangoes.load_Anderson_data()
NIR 306
NIR_truncated 103
SNV 103
SG1 103
SG2 103
SNV_SG1 103
SNV_SG2 103
Total number of features:  924
```

The data is saved a big DataFrame:
```python

>>> d['data_dict']['shuffled_df']
       index     Set  Season Region        Date        Type Cultivar  Pop  ... SNV_SG2_972  SNV_SG2_975  SNV_SG2_978  SNV_SG2_981  SNV_SG2_984  SNV_SG2_987  SNV_SG2_990  ignore
0       8987  Tuning       3     NT   9/10/2017  Hard Green       KP   60  ...   -0.009409    -0.009409    -0.009409    -0.009409    -0.009409    -0.009409    -0.009409   False
1       8500  Tuning       3     NT  14/08/2017  Hard Green     Caly   45  ...   -0.010544    -0.010544    -0.010544    -0.010544    -0.010544    -0.010544    -0.010544   False
2       5489     Cal       3     NT  23/10/2017  Hard Green     1243   65  ...   -0.009629    -0.009629    -0.009629    -0.009629    -0.009629    -0.009629    -0.009629   False
3        651     Cal       1     NT  23/10/2015  Hard Green       KP    5  ...   -0.010373    -0.010373    -0.010373    -0.010373    -0.010373    -0.010373    -0.010373   False
4       4552     Cal       3     NT  27/09/2017  Hard Green    LadyG   52  ...   -0.009183    -0.009183    -0.009183    -0.009183    -0.009183    -0.009183    -0.009183   False
...      ...     ...     ...    ...         ...         ...      ...  ...  ...         ...          ...          ...          ...          ...          ...          ...     ...
11686   7299     Cal       3    QLD  26/01/2018       Ripen       HG   92  ...   -0.013949    -0.013949    -0.013949    -0.013949    -0.013949    -0.013949    -0.013949   False
11687   3086     Cal       1    QLD   1/02/2016  Hard Green    Keitt   24  ...   -0.006679    -0.006679    -0.006679    -0.006679    -0.006679    -0.006679    -0.006679   False
11688   6753     Cal       3    QLD   4/01/2018       Ripen     R2E2   83  ...   -0.017772    -0.017772    -0.017772    -0.017772    -0.017772    -0.017772    -0.017772   False
11689   1474     Cal       1     NT  23/10/2015  Hard Green       KP    8  ...   -0.009754    -0.009754    -0.009754    -0.009754    -0.009754    -0.009754    -0.009754   False
11690   8603  Tuning       3     NT  27/09/2017  Hard Green     1201   48  ...   -0.009085    -0.009085    -0.009085    -0.009085    -0.009085    -0.009085    -0.009085   False

[11691 rows x 835 columns]
```

The columns corresponding to NIR spectra can be find in the `d['feature_columns']['NIR']` variable. For example:
```python
NIR_columns = d['feature_columns']['NIR']
NIR_spectra = df[NIR_columns]
```

And other pre-processed variants are available which we can see by inspecting the `d['feature_columns']` dictionary:
```python
>>> d['feature_columns'].keys()
odict_keys(['NIR', 'NIR_truncated', 'SNV', 'SG1', 'SG2', 'SNV_SG1', 'SNV_SG2'])
```

# How to train the models
All the example commands below utilize the mangoes dataset described above. We recommend verifying that these work prior to customizing with your own dataset.

## Predict the Average (PtA)

This is a "dummy" model that just "predicts" the average value of the target variable from the training set. The command below trains the model on each fold of cross-validation and saves the results to a new directory created within the `resultsDir` (replace this with your own path):

```bash
python train_model.py SKLEARN --out_dir_naming AUTO --resultsDir "C:/model_results" --fold_spec "{'type': '10fold_and_test_split', 'use_dev':False}" --scaler_settings "{'X_type':'none','Y_type':'none'}" --input_features "['NIR_truncated']" --dataset_name mangoes --which_targets_set DM --model_name predict_the_average --m Mangoes_PtA --which_folds "[0,1,2,3,4,5,6,7,8,9]"
```

### Description of command line arguments for `train_model.py`:
- `m`: Short for "message"; this is any name or message you want, it is appended to the output directory of the results.
- `out_dir_naming`: If set to "AUTO", an output directory will be created with a unique ID followed by `m`. If set to "MANUAL", the output directory name will be `m` (you must make sure it doesn't already exist).
- `resultsDir`: Directory where output directories will be created for the results.
- `fold_spec`: dictionary describing the type of cross-validation or training/test-set split.
- `scaler_settings`: sets what type of scaling to apply to the input data (`X_type`) or to the target variable (`Y_type`); valid types are `none`, `min_max`, or `mean_std`.
- `input_features`: list of names, where each name is a collection of features. i.e. names of blocks or sensor types.
- `dataset_name`: name of the dataset to use
- `which_targets_set`: name of the prediction target variable (or set of variables, if any sets are defined)
- `model_name`: name of the regression model to use (valid options are defined in `train_sklearn_model.py`)
- `which_folds`: list of folds of cross-validation

## PLS

PLS model with 5 components is trained by setting the `model_params` argument to `{'n_components':5}` (surrounded by quotes so that the shell passes it in properly) like so:
```bash
python train_model.py SKLEARN --out_dir_naming AUTO --resultsDir C:/model_results --fold_spec "{'type': '10fold_and_test_split', 'use_dev': False}" --scaler_settings "{'X_type':'none','Y_type':'mean_std'}" --input_features "['NIR_truncated']" --dataset_name mangoes --which_targets_set DM --model_name PLS --model_params "{'n_components':5}" --m "Mangoes_PLS,NIR,n_comp=5" --which_folds "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
```

For the purpose of demonstrating code functionality,
we'll use NIR and the first sensor and the 1st derivative normalized by SNV as the 2nd "sensor", which is named `SNV_SG1`. The PLS model in this case in trained by changing the `input_features` argument:
```bash
python train_model.py SKLEARN --out_dir_naming AUTO --resultsDir C:/model_results --fold_spec "{'type': '10fold_and_test_split', 'use_dev': False}" --scaler_settings "{'X_type':'none','Y_type':'mean_std'}" --input_features "['SNV_SG1']" --dataset_name mangoes --which_targets_set DM --model_name PLS --model_params "{'n_components':5}" --m "Mangoes_PLS,SNV_SG1,n_comp=5" --which_folds "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
```

## High-level sensor fusion: with NNLS

High-level sensor fusion is accomplished by taking the predictions from 2 or more existing models (representing different sensors or blocks). Using the two PLS models trained in the examples, the sensor fusion command is:
```bash
python train_LS_sensor_fusion.py --target DM --m DM_highlevel_fusion --prefix "PLS+" --input_dirs "[{'dir_path':'C:/model_results/[000029]~Mangoes_PLS,NIR,n_comp=5','nickname':'NIR'},{'dir_path':'C:/model_results/[000057]~Mangoes_PLS,SNV_SG1,n_comp=5','nickname':'SNV_SG1'}]"
```

### Description of command line arguments for `train_LS_sensor_fusion.py`:
- `target`: name of the prediction target
- `m`: Short for "message"; this is any name or message you want, it is appended to the output directory of the results.
- `prefix`: A short prefix (anything you want), that describes the input models (in the above example, we're fusing PLS models)
- `input_dirs`: List of dictionaries describing the input directories for the sensors or blocks you want to do sensor fusion on. e.g., you could put a PLS model that used NIR spectra as `{'dir_path': PATH_HERE, 'nickname': 'NIR'}` where `PATH_HERE` is the output directory created by `train_model.py SKLEARN --model_name PLS ...`

## High-level sensor fusion: with ROSA

High-level sensor fusion using ROSA:
```bash
python train_model.py ROSA --out_dir_naming AUTO --resultsDir C:/model_results --fold_spec "{'type': '10fold_and_test_split', 'use_dev': False}" --scaler_settings "{'X_type':'none','Y_type':'mean_std'}" --input_features "['NIR_truncated','SNV','SNV_SG1','SNV_SG2']" --dataset_name mangoes --which_targets_set DM --n_components 40 --m Mangoes_ROSA,n_comp=40 --which_folds "[0,1,2,3,4,5,6,7,8,9]"
```
In this example, we use NIR and some pre-processed variants of NIR as additional blocks in the model (these are analogous to having different sensors), which are specified in the `input_features` argument.


## Neural networks

Neural networks are trained using `python train_model.py NN ...`. Command-line arguments may be used to specify the model architecture and hyperparameters. However, a configuration file can also be used (if both command-line arguments and a file are used, the command-line arguments take precedence) by using the `cmd_args_fpath` command-line argument to specify the path to the config file.

For example, a CNN model with 1 hidden layer (2 fully-connected layers) is specified in [CNN_for_mangoes.pyon](spectra_ml/example_NN_models/CNN_for_mangoes.pyon) which contains:

	{
	  'out_dir_naming': 'AUTO',
	  'm': 'Mangoes_CNN',
	  'n_in_parallel': 0, 
	  'fold_spec': {'type': '10fold_and_test_split', 'use_dev': True},
	  'scaler_settings': {
	    'X_type': 'mean_std',
	    'Y_type': 'mean_std',
	  },
	  'dataset_name': 'mangoes',
	  'do_ES': True,
	  'n_training_runs': 40,
	  'which_folds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	  'which_targets_set': 'DM',
	  'base_LR': 1e-06,
	  'conv_filter_width': 11,
	  'batch_size': 128,
	  'n_full_epochs': 10000,
	  'LR_sched_settings': {'type': 'ReduceLROnPlateau', 'patience': 25, 'factor': 0.5, 'base_min_LR': 1e-08},
	  'ES_patience': 50,
	  'input_features': ['NIR_truncated'],
	  'FC_L2_reg_factor': 0.01,
	  'FC_init': 'he_normal',
	  'conv_L2_reg_factor': 0.01,
	  'conv_filter_init': 'he_normal',
	  'conv_n_filters': 10,
	  'FC_size_per_input': [8],
	  'FC_L2_reg_factor_per_input': [0.01],
	}

Train this model by calling
```bash
python train_model.py NN --cmd_args_fpath "./example_NN_models/CNN_for_mangoes.pyon" --n_in_parallel 3
```
here we override the `n_in_parallel` argument to set it to train 3 models in parallel. You can set it to whatever value is appropriate for your computer's CPU and GPU resources.

Similarly, a sensor fusion model is specified in [sensor_fusion_for_mangoes.pyon](spectra_ml/example_NN_models/sensor_fusion_for_mangoes.pyon). For sensor fusion models, note that the following arguments must provide a list that has one value per sensor:
- `conv_n_filters`: number of conv filters, one per sensor (sensors specified in `input_features`)
- `conv_filter_width`: width of the conv filters, one per sensor
- `FC_size_per_input`: number of units in the fully-connected layer, one per sensor
- `FC_L2_reg_factor_per_input`: L2 regularization factor, one per sensor

The model is trained like so:
```bash
python train_model.py NN --cmd_args_fpath "./example_NN_models/sensor_fusion_for_mangoes.pyon"
```

## Config file

Additional settings are available in `config.toml` (which is created after running any `train_model` model at least once). The settings allow:
- setting a default for the results directory 
- toggling whether to plot each of the various figures
- toggling whether to save checkpoints

## Hyperparameter Optimization (HPO)

Neural network hyperparameter requires [hyperopt](https://github.com/hyperopt/hyperopt) version `0.2.5`.

Performing HPO with hyperopt requires running a "master" process which maintains a MongoDB database of hyperparameter configurations (known as trials) that have been tried so far
and decides what configuration to try next.
Then, one or more "workers" reads what configuration to try next, trains a neural network using this configuration, and reports the validation score back to the "master".

For more details, see documentation on hyperopt (http://hyperopt.github.io/hyperopt/) and [this page](http://hyperopt.github.io/hyperopt/scaleout/mongodb/) in particular.
In this project, wrapper scripts have been written to aid in launching the master and workers, explained next.

### Master

Master process is started as follows. Square brackets (`[--abc def]`) denote optional arguments, angle brackets (`<abc>`) are required.

```bash
python HPO_start_master.py <database_name> <max_evals> [--DB_host 127.0.0.1] [--DB_port 27017] [--out_dir .] --which_cmd_space <which_cmd_space> --target=<target>
```

Three hyperparameter search spaces are included in this repository, corresponding to NN1, NN2, and NN3 (sensor fusion) described in the paper. NN1 and NN2 are defined [here](spectra_ml/components/hyperopt_config/cmd_space_for_mangoes.py) and NN3 is defined [here](spectra_ml/components/hyperopt_config/cmd_space_for_mangoes_sensor_fusion.py).

The master processes are launched in background like so:

```bash
nohup python HPO_start_master.py "HPO_01_MANGOES_NN1" 1100 --out_dir=/home/ubuntu/ --which_cmd_space=MANGOES_NN1 --target=DM &> ~/output_HPO_01.txt &
nohup python HPO_start_master.py "HPO_02_MANGOES_NN2" 1100 --out_dir=/home/ubuntu/ --which_cmd_space=MANGOES_NN2 --target=DM &> ~/output_HPO_02.txt &
nohup python HPO_start_master.py "HPO_03_MANGOES_NN3" 1100 --out_dir=/home/ubuntu/ --which_cmd_space=MANGOES_NN3 --target=DM &> ~/output_HPO_03.txt &
```


### Workers

Once the "master" is running. Run one or more workers. Workers must have access to the database on "master".
The worker process is started as follows:

```bash
python HPO_hyperopt_mongo_worker.py <database_name> [--DB_host 127.0.0.1] [--DB_port 27017] [--n_jobs 9999999] [--timeout_hours None]
```

We used SSH port forwarding between an Ubuntu server and compute cluster worker nodes, 
but in this example we assume a worker running on the same node as the master:

```bash
#!/bin/bash
database_name=HPO_01_MANGOES_NN1
LOCALPORT=27017
python -u HPO_hyperopt_mongo_worker.py $database_name "127.0.0.1" $LOCALPORT --timeout_hours 11
```

This process runs for 11 hours, then quits after completing the next trial.
The `-u` flag to python "forces the stdout and stderr streams to be unbuffered" which forces outputs to the screen right away; this is useful when trying to debug problems.

Alternatively, on a Windows machine, you may use a `.bat` script with the following
```bat
@echo off
setlocal

cd git-repo\spectra_ml

set OVERRIDE_n_gpu=1
set OVERRIDE_n_in_parallel=0

python HPO_hyperopt_mongo_worker.py "HPO_01_MANGOES_NN1" --DB_host 127.0.0.1 --DB_port 9017 --n_jobs 1
```
This (optionally) overrides the `n_gpu` and `n_in_parallel` arguments for this worker (useful for testing).

Once workers have completed some or all the trials, the MongoDB database will contains all the trials' details, including RMSECV score which may be minimized to find the best-performing hyperparameter configuration. For more details on hyperopt, [this tutorial](https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt/notebook) is very thorough.

# If you have questions...

Firstly, please be sure to read the paper (you can find a link [here](https://skylogic.ca/)). Secondly, feel free to ask a question by posting in the "Issues" section on github! 