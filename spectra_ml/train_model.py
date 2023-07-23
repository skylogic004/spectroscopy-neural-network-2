import fire
import os
from os.path import join, split, realpath, dirname, splitext, isfile, exists, isdir
from os import listdir
from colorama import Fore, Back, Style
import colorama
import spectra_ml.components.colorama_helpers as _c
import pprint
import inspect
from datetime import datetime as dt
from timeit import default_timer as timer
import socket
import logging
import subprocess
from sklearn.metrics import mean_squared_error
import json
import pandas as pd
import numpy as np
import sys
APP_DIR = split(realpath(__file__))[0]
sys.path.append(APP_DIR)

from spectra_ml.components.cmd_line_helper import merge_args_with_args_in_file
from spectra_ml.components.config_io.config_toml import getConfig, getConfigOrCopyDefaultFile
from spectra_ml.components.get_results_dir import get_results_dir
from spectra_ml.components.prepare_out_dir_and_logging import prepare_out_dir_and_logging #, cleanup_tmp_link
from spectra_ml.components.data_loader import generic_loader

__author__ = "Matthew Dirks"


IS_LINUX = os.name == 'posix'
'''
UUID = uuid.uuid4().hex
if (IS_LINUX):
	SCRATCH_PATH = os.environ.get('SCRATCH') # set on ComputeCanada cluster
	if (SCRATCH_PATH is not None):
		SYMLINK_FPATH = join(SCRATCH_PATH, 'tmp_symlink_to_results_'+UUID)
	else:
		SYMLINK_FPATH = os.path.expanduser('~/tmp_symlink_to_results_'+UUID)
else: # is Windows OS
	SYMLINK_FPATH = None

def path_for_tf(fpath, out_dir):
	# tensorflow is sensitive to special characters (on linux; windows is fine),
	# and I often use square brackets in my directory names,
	# so this makes a symlink to the output directory so TF doesnt see the special characters.
	# This function swaps the original output directory with the symlink:
	if (IS_LINUX):
		return fpath.replace(out_dir, SYMLINK_FPATH)
	else:
		return fpath
'''

COMPUTER_NAME = socket.gethostname()

CACHE_DIR = join(APP_DIR, 'data')
CONFIG_FPATH = "config.toml"
CONFIG_DEFAULT_FPATH = "config_DEFAULT.toml"

rmse = lambda arr1, arr2: np.sqrt(mean_squared_error(arr1, arr2))

def get_git_revision_hash():
	try:
		return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
	except Exception as e:
		return f'Couldn\'t get git revision. Error: {e}'


def main_inside(which_program,
	#=== THESE COME DIRECTLY FROM COMMAND LINE (OR OVERRIDDEN BY CMD_ARGS_FPATH) === 
		resultsDir=None,
		m=None,
		out_dir_naming='AUTO', # alternative: MANUAL which uses `m` as output directory name
		dataset_name='Li_hole1',
		which_folds=[0],
		# fold_spec={'type':'rand_split'},
		# scaler_settings={'X_type': 'none', 'Y_type': 'none'},
		# input_features=['XRF'],
		which_targets_set='ALL',
		**new_kwargs
	):
	assert 'kth_fold' not in new_kwargs, 'Use `which_folds` instead (and provide a list)'

	############################# SETTING UP ###############################
	start_datetime = dt.now() # for printing total time in nice human-readable format
	start_time = timer() # for getting total time in seconds
	prev_time = timer()

	# dump parameters
	f_args, _, _, f_values = inspect.getargvalues(inspect.currentframe())
	cmd_args_dict = {k:v for k, v in f_values.items() if k in f_args}
	cmd_args_dict.update(new_kwargs)
	print('Command args: {}'.format(pprint.pformat(cmd_args_dict, width=100)))

	# get configuration settings
	configData, _ = getConfigOrCopyDefaultFile(CONFIG_FPATH, CONFIG_DEFAULT_FPATH)
	resultsDir = get_results_dir(configData, resultsDir)


	################### create output sub-directory within results directory ############################
	if (out_dir_naming == 'AUTO'):
		use_task_run_id = True
	elif (out_dir_naming == 'MANUAL'):
		use_task_run_id = False
	else:
		raise(ValueError('Invalid out_dir_naming'))

	out_dir, task_run_id = prepare_out_dir_and_logging(
		# SYMLINK_FPATH, 
		resultsDir=resultsDir, 
		comment_message=m,
		cmd_args_dict=cmd_args_dict,
		global_id_fpath=None,
		fallback_resultsDir=configData['paths']['results_dir'],
		use_task_run_id=use_task_run_id,
	)
	logger = logging.getLogger('spectra_ml')

	############################# PRINT INFO ###############################

	logger.info(_c.m_green('\n===DEBUG INFO==='))
	logger.info(f'COMPUTER_NAME: {COMPUTER_NAME}')
	logger.info(f'task_run_id: {task_run_id}')
	logger.info(f'out_dir: {out_dir}')
	# logger.info(f'SYMLINK_FPATH: {SYMLINK_FPATH}')
	logger.info(f'git revision: {get_git_revision_hash()}')
	logger.info(f'cwd: {os.getcwd()}')

	logger.info('Environment variable SLURM_ARRAY_TASK_ID: {}'.format(os.environ.get('SLURM_ARRAY_TASK_ID')))
	logger.info('Environment variable SLURM_TMPDIR: {}'.format(os.environ.get('SLURM_TMPDIR')))

	logger.info(_c.m_green('\n===CONFIG DATA==='))
	logger.info('configData = ' + str(configData))


	############################ LOAD DATA ####################################
	logger.info(_c.m_green('\n===LOAD DATA==='))
		
	if (dataset_name in generic_loader.EXPECTED_DATASET_NAMES):
		dataset_dict = generic_loader.load(dataset_name, CACHE_DIR)
	else:
		raise(ValueError('Invalid dataset_name'))

	# cur_data = dataset_dict['cur_data']
	target_columns_dict = dataset_dict['target_columns_dict']

	# setup which_targets to target
	if (which_targets_set == 'ALL'):
		which_targets = [el for el in target_columns_dict.keys()]
	else: # specified ONE target
		which_targets = [which_targets_set]

	# get column name for each target (sometimes these are different, sometimes they are the same)
	print('which_targets=',which_targets,'target_columns_dict=',target_columns_dict)
	target_columns_in_use = [target_columns_dict[target] for target in which_targets]
	logger.info('which_targets (n={}) = {}'.format(len(which_targets), which_targets))
	logger.info('target_columns_in_use (n={}) = {}'.format(len(target_columns_in_use), target_columns_in_use))


	############################# UPDATE KWARGS ###############################
	new_kwargs['configData'] = configData
	new_kwargs['start_datetime'] = start_datetime
	new_kwargs['dataset_dict'] = dataset_dict
	new_kwargs['target_columns_in_use'] = target_columns_in_use

	# remove these:
	for key in ['resultsDir','m','out_dir_naming','dataset_name','which_targets_set']:
		new_kwargs.pop(key, None)

	############################# PASS ARGS TO PROGRAM THAT TRAINS REQUESTED MODEL ###############################
	logger.info(_c.m_green(f'\n===RUNNING training procedure for {which_program} on fold(s): {which_folds} ==='))
	if (which_program == 'NN'):
		from spectra_ml.components.models.train_TF_model import train_TF_model
		train_function = train_TF_model
	elif (which_program == 'SKLEARN'):
		from spectra_ml.components.models.train_sklearn_model import train_sklearn_model
		train_function = train_sklearn_model
	elif (which_program == 'ROSA'):
		from spectra_ml.components.models.train_ROSA_model import train_ROSA_model
		train_function = train_ROSA_model
	else:
		logger.error(_c.m_warn2('Invalid model name.'))

	pred_per_fold = []
	for kth_fold in which_folds:
		logger.info(_c.m_green(f'\n===RUNNING fold {kth_fold} ==='))
		new_kwargs['out_dir'] = join(out_dir, f'k={kth_fold}')
		os.makedirs(new_kwargs['out_dir'])
		new_kwargs['kth_fold'] = kth_fold
		predictions_dict = train_function(**new_kwargs)

		# save for later
		pred_per_fold.append(predictions_dict)

	############################# AGGREGATE SCORE ACROSS FOLDS ###############################
	get_test_CV_samples = lambda df: df[df['set']=='test_CV']
	test_CV_dfs = [get_test_CV_samples(d['df']) for d in pred_per_fold]
	test_CV_df = pd.concat(test_CV_dfs, axis=0)
	assert len(test_CV_df.index.unique()) == test_CV_df.shape[0], 'Shouldn\'t be any duplicate samples'
	logger.info(f'After combining data across folds, there are {test_CV_df.shape[0]} samples')

	prediction_columns = pred_per_fold[0]['prediction_columns']
	target_columns = pred_per_fold[0]['target_columns']

	result_info = {
		'which_folds': which_folds,
		'target_columns': target_columns,
	}

	for target, pred in zip(target_columns, prediction_columns):
		result_info[f'RMSECV_{target}'] = rmse(test_CV_df[target], test_CV_df[pred])

	with open(join(out_dir, 'result_info.json'), 'w') as text_file:
		text_file.write(json.dumps(result_info, indent=2))

	############################# FINISHING UP ###############################

	# success sound
	# THIS BLOCKS OCCASIONALLY - NEED A TIMEOUT
	# if (COMPUTER_NAME == 'MD-X'):
	# 	try:
	# 		import beepy
	# 		beepy.beep('ready')
	# 		print('(played notification sound)')
	# 	except:
	# 		pass
	# tmp_datetime = dt.now()
	# logger.info(f'=== notification sound DONE === (started at {start_datetime}, now is {tmp_datetime})')

	# remove symlink, if needed
	# cleanup_tmp_link(SYMLINK_FPATH)

	# print time spent
	end_datetime = dt.now()
	logger.info('Program finished! (started at {}, ended at {}, time spent: {} (hh:mm:ss.micro)'.format(start_datetime, end_datetime, end_datetime - start_datetime))

	print('DONE train_model.py')

def main(which_program, *args, **kwargs):
	print(f'=== which_program = {which_program}, num positional args: {len(args)}, num kw args: {len(kwargs)} ===')

	# load config file from `cmd_args_fpath` keyword argument.
	# Those settings will be overridden by any specified in `kwargs`
	new_kwargs = merge_args_with_args_in_file(*args, **kwargs)

	main_inside(which_program, **new_kwargs)

if __name__ == '__main__':
	colorama.init(wrap=False)
	fire.Fire(main)