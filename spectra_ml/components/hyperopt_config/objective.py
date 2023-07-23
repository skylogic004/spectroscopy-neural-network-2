"""
This takes a sample of the hyperparameter space (which I call a "cmd_space") and evaluates it. 
`HPO_start_master.py` registers `calc_loss` as the function for minimization.
"""
from subprocess import Popen, PIPE, STDOUT
import uuid
import pandas as pd
import os
from os.path import join, exists, isdir
from os import listdir, makedirs
from hyperopt import STATUS_OK, STATUS_FAIL
from shutil import copyfile
import numpy as np
import socket
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer
import signal
import time
import json

from spectra_ml.components.hyperopt_config.make_cmds import build_cmd
from spectra_ml.components.hyperopt_config.helpers import process_sample
from spectra_ml.components.config_io.config_toml import getConfig, getConfigOrCopyDefaultFile
from spectra_ml.components.get_results_dir import get_results_dir
from train_model import CONFIG_FPATH, CONFIG_DEFAULT_FPATH, APP_DIR

try:
	from arca2.mypushbullet import notify
except:
	# if notify function unavilable, do nothing
	def notify(*_): pass

__author__ = "Matthew Dirks"

IS_LINUX = os.name == 'posix'
WINDOWS_MAX_LENGTH = 8190
COMPUTER_NAME = socket.gethostname()

REQUIRED_NUM_ENSEMBLE_RUNS = 40
# REQUIRED_NUM_ENSEMBLE_RUNS = 1 # for testing

# If a training run ended too quickly (i.e., time < WARNING_TIME_THRESHOLD),
# it's quite likely than error occurred, so give a warning and sleep for SLEEP_TIME
WARNING_TIME_THRESHOLD = 60*20 # 20 minutes - this is better for sensor fusion and bigger models
# WARNING_TIME_THRESHOLD = 300 # better for small single-sensor models
SLEEP_TIME = 300 # seconds

def calc_RMSE(arr1, arr2):
	try:
		return np.sqrt(mean_squared_error(arr1, arr2))
	except ValueError: # sometimes I see "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')."
		return None

BEST_MODEL_PRED_FNAME = 'best_model_predictions.pkl'
RESULT_INFO_FNAME = 'result_info.json'
CONSOLE_OUT_FNAME = 'consoleOutput.txt'

def calc_loss(sample_of_cmd_space, hyperhyperparams):
	print('\n=============== STARTING objective.calc_loss =============== ')
	st = timer()

	# prepare dictionary to output the results
	return_dict = {
		'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID'),
		'status': STATUS_OK, # this is overwritten if something goes wrong
	}

	results_dir_kind = 'NOT SET YET'

	SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR') # set on any SLURM HPC cluster
	SCRATCH_PATH = os.environ.get('SCRATCH') # set on ComputeCanada cluster
	if (SLURM_TMPDIR is not None):
		results_dir_for_HPO_trials = SLURM_TMPDIR
		results_dir_kind = 'SLURM_TMPDIR'
	else: 
		# get path to results_dir from config file
		configData, _ = getConfigOrCopyDefaultFile(join(APP_DIR, CONFIG_FPATH), join(APP_DIR, CONFIG_DEFAULT_FPATH))
		parent_results_dir = get_results_dir(configData)
		assert exists(parent_results_dir)

		results_dir_for_HPO_trials = join(parent_results_dir, 'HPO', hyperhyperparams['target'], hyperhyperparams['which_cmd_space'])
		results_dir_kind = 'results_dir_from_config'
		if (not exists(results_dir_for_HPO_trials)):
			makedirs(results_dir_for_HPO_trials)

	assert exists(results_dir_for_HPO_trials), f'results_dir_for_HPO_trials ({results_dir_for_HPO_trials}) doesnt exist'

	
	print('objective.py: CWD={}'.format(os.getcwd()))

	######## DO THE WORK TO COMPUTE RESULTS FOR THIS TRIAL
	UUID = uuid.uuid4().hex
	return_dict = do_the_work(sample_of_cmd_space, hyperhyperparams, UUID, results_dir_for_HPO_trials, return_dict)

	######## DONE PROCESSING
	# record time taken
	tt = timer()-st
	return_dict['time_duration_seconds'] = tt
	if (tt > 3600):
		print(f'time duration = {tt:0.0f} = {tt/3600:0.2f} hours')
	else:
		print(f'time duration = {tt:0.0f} = {tt/60:0.2f} minutes')

	# now that processing is done, lets copy files back to SCRATCH for storage (if running on a SLURM cluster)
	if (results_dir_kind == 'SLURM_TMPDIR' and SCRATCH_PATH is not None):
		copy_logs_to_scratch(return_dict['task_dir_path'], join(SCRATCH_PATH, 'HPO', hyperhyperparams['target'], hyperhyperparams['which_cmd_space'], f'TASKDIR_{UUID}'))

	print('=============== END OF objective.calc_loss =============== ')


	return return_dict

def do_the_work(sample_of_cmd_space, hyperhyperparams, UUID, results_dir, return_dict):
	""" This function does the following:
	- take the `sample_of_cmd_space` (i.e. a "trial" which is a set of hyperparameter values)
	- generate the command needed to run NN training given the hyperparameters
	- run the NN training command
	- read the results
	- return "loss" which is the metric that hyperopt optimizes
	"""

	# prepare
	return_dict['UUID'] = UUID
	return_dict['outputs'] = {}
	return_dict['cmds'] = []
	return_dict['error_messages'] = []
	return_dict['n_gpu'] = os.environ.get('OVERRIDE_n_gpu')
	return_dict['n_in_parallel'] = os.environ.get('OVERRIDE_n_in_parallel')

	################################ post-process sample
	cmd_dict = process_sample(sample_of_cmd_space)
	cmd_dict['resultsDir'] = results_dir

	################################ generate and run command
	task_dir_name = f'TASKDIR_{UUID}'
	cmd_dict['m'] = task_dir_name
	cmd_dict['out_dir_naming'] = 'MANUAL'
	task_dir_path = join(cmd_dict['resultsDir'], task_dir_name)
	return_dict['task_dir_path'] = task_dir_path

	for key in ['n_gpu']:
		cmd_dict[key] = os.environ.get(f'OVERRIDE_{key}')
		assert cmd_dict[key] is not None
		print(f'Retrieved from environment var: {key} = {cmd_dict[key]}')

	n_in_parallel = os.environ.get('OVERRIDE_n_in_parallel', 0)
	cmd_dict['n_in_parallel'] = n_in_parallel

	cmd_ser = pd.Series(cmd_dict)
	cmd_str = build_cmd(None, cmd_ser)
	return_dict['cmds'].append(cmd_str)

	# check length (on Windows we run into issues if too long)
	if (len(cmd_str) > WINDOWS_MAX_LENGTH and not IS_LINUX):
		msg = f'cmd_str too long for windows (limit is probably {WINDOWS_MAX_LENGTH} and this length is {len(cmd_str)})'
		print(msg)
		return_dict['error_messages'].append(msg)

	# run command to train NN
	print(f'---------> calling cmd {cmd_str}')
	st = timer()
	p = Popen(cmd_str, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd='..');

	# save stdout (ONLY the stdout, NOT what's printed via logger.info(...) etc)
	_stdout = p.stdout.read().decode('UTF-8') # blocks until done
	training_time = timer() - st
	return_dict['outputs'][f'task_stdout'] = '{}\n...\n{}'.format(_stdout[:800], _stdout[-500:]) # only some chars b/c if too big mongo dies
	
	# print stdout for debugging purposes (only if time was less than threshold)
	if (training_time < WARNING_TIME_THRESHOLD):
		print('---------> stdout:')
		print(_stdout)
		print('---------> END OF stdout')

	# safety check: if training took less than WARNING_TIME_THRESHOLD then probably something is broken.
	#               So, rather than continue failing repeatedly, lets sleep for a while (until I notice and fix it)
	if (training_time < WARNING_TIME_THRESHOLD):
		msg = f'WARNING: training finished suspiciously fast ({training_time:0.1f} < {WARNING_TIME_THRESHOLD} s), sleeping now for {SLEEP_TIME} seconds...'
		print(msg)

		# notify me of an issue to my phone:
		notify('PYTHON objective.py', msg + f'task_dir_name={task_dir_name} @{COMPUTER_NAME}')

		time.sleep(SLEEP_TIME)

	### Load consoleOutput.txt, save last N lines of text
	fp = join(task_dir_path, CONSOLE_OUT_FNAME)
	if (exists(fp)):
		with open(fp, 'r') as f:
			return_dict['outputs'][f'task_log'] = ''.join(f.readlines()[-15:])

	########################## Ensembles: check that required number of models is completed ###################################################
	for kth_fold in cmd_dict['which_folds']:
		fp = join(task_dir_path, f'k={kth_fold}', BEST_MODEL_PRED_FNAME)

		if (exists(fp)):
			pkl_data = pd.read_pickle(fp)
			edf = pkl_data['ensemble_runs_df']

			assert len(pkl_data['prediction_columns'])==1, 'This code assumes there is only 1 target (output) from the NN'
			pred_target = pkl_data['prediction_columns'][0] # e.g. 'DM_pred'
			target = pkl_data['target_columns'][0] # e.g. 'DM'

			# get names of the columns which have the predictions for each ensemble-model (i.e. each run)
			# e.g. "run0:DM_pred", ..., "run39:DM_pred"
			per_run_pred_columns = sorted([x for x in edf.columns if x.endswith(pred_target)])

			for _set, set_df in edf.groupby('set', sort=False):
				# calc and store the RMSE of each training run. Useful in plotting variance in post-analysis
				RMSE_scores = []
				for col in per_run_pred_columns:
					RMSE_scores.append(calc_RMSE(set_df[target], set_df[col]))

				return_dict[f'{target}_k={kth_fold}_{_set}_RMSE_per_training_run'] = RMSE_scores

			# how many models does the ensemble have?
			num_training_runs = len(per_run_pred_columns)
			
			# verify number of training runs
			if (num_training_runs < REQUIRED_NUM_ENSEMBLE_RUNS):
				return_dict = abort(return_dict, f'Ensemble collected only {num_training_runs} runs, but at least {REQUIRED_NUM_ENSEMBLE_RUNS} are required.')
				return return_dict
		else:
			return_dict = abort(return_dict, f'BEST_MODEL_PRED_FNAME FILE_NOT_FOUND ({fp}); Ensemble requires this.')
			return return_dict
		

	################################ read in scores per fold
	result_info_per_fold = []
	for kth_fold in cmd_dict['which_folds']:
		fp = join(task_dir_path, f'k={kth_fold}', RESULT_INFO_FNAME)
		if (exists(fp)):
			with open(fp, 'r') as f:
				result_info = json.load(f)
				result_info_per_fold.append(result_info)
		else:
			msg = f'RESULT_INFO_FNAME FILE_NOT_FOUND ({fp})'
			print(msg)
			return_dict['error_messages'].append(msg)
			return_dict[join(f'k={kth_fold}', RESULT_INFO_FNAME)] = 'FILE_NOT_FOUND'

	# save to database
	return_dict['result_info_per_fold'] = result_info_per_fold


	################################  record loss value
	# get RMSECV (the score across all folds)
	fp = join(task_dir_path, RESULT_INFO_FNAME)
	if (exists(fp)):
		with open(fp, 'r') as f:
			final_result_info = json.load(f)

		# save to database
		return_dict.update(final_result_info)

		target_columns = final_result_info['target_columns']
		assert len(target_columns)==1
		target = target_columns[0]
		loss_var_name = f'RMSECV_{target}'

		return_dict['loss'] = final_result_info[loss_var_name]
	else:
		msg = f'RESULT_INFO_FNAME FILE_NOT_FOUND ({fp})'
		print(msg)
		return_dict['error_messages'].append(msg)
		return_dict[RESULT_INFO_FNAME] = 'FILE_NOT_FOUND'

		return_dict = abort(return_dict, f'No loss data to record because RESULT_INFO_FNAME missing')
		return return_dict

	# return successful results
	return return_dict



def abort(return_dict, reason_message):
	""" If aborting due to error, pass in the message here.
	Returns updated return_dict
	"""
	txt = f'Aborted (and status set to STATUS_FAIL); {reason_message}'
	print(txt)
	return_dict['status'] = STATUS_FAIL
	return_dict['reason_for_failure'] = txt
	return return_dict



KEEP_THESE_FILES = [CONSOLE_OUT_FNAME, 
                    'cmd_args.pyon',
                    'cmd.txt',
                    RESULT_INFO_FNAME]
KEEP_THESE_FILES_PER_FOLD = ['run_00/loss_history.png',
                             'run_00/conv_filters.png',
                             'prediction_plots/predictions_Li_%_(ME-MS61)_test_CV.png',
                             'prediction_plots/predictions_Zr_%_(ME-MS61,Zr-XRF10)_test_CV.png',
                             'prediction_plots/predictions_Mg_%_(ME-MS61)_test_CV.png',
                             'prediction_plots/predictions_Rb_%_(ME-MS61)_test_CV.png',
                             BEST_MODEL_PRED_FNAME,
                             RESULT_INFO_FNAME]
KEEP_THESE_FILES_PER_RUN = ['saved_checkpoint.data-00000-of-00001', 'saved_checkpoint.index']

def copy_logs_to_scratch(task_dir_path, outDir):
	if (not exists(outDir)):
		os.makedirs(outDir)

		for keep_fname in KEEP_THESE_FILES:
			fp = join(task_dir_path, keep_fname)
			if (exists(fp)):
				print(f'Copying {fp}')
				new_fname = keep_fname.replace('/', '--')
				copyfile(fp, join(outDir, new_fname))

		for k_subdir in [x for x in listdir(task_dir_path) if isdir(join(task_dir_path,x))]:
			for keep_fname in KEEP_THESE_FILES_PER_FOLD:
				fp = join(task_dir_path, k_subdir, keep_fname)
				if (exists(fp)):
					print(f'Copying {fp}')
					new_fname = k_subdir + '--' + keep_fname.replace('/', '--')
					copyfile(fp, join(outDir, new_fname))

			for run_subdir in [x for x in listdir(join(task_dir_path, k_subdir)) if isdir(join(task_dir_path, k_subdir, x)) and 'run' in x]:
				run_subdir_path = join(task_dir_path, k_subdir, run_subdir)
				for keep_fname in KEEP_THESE_FILES_PER_RUN:
					fp = join(run_subdir_path, keep_fname)
					if (exists(fp)):
						new_fname = join(k_subdir, run_subdir, keep_fname).replace('/', '--')
						copyfile(fp, join(outDir, new_fname))


		# keep any ensemble training run outputs (e.g. training_run_000.pkl)
		# fnames = [fname for fname in listdir(task_dir_path) if fname.startswith('training_run_')]
		# for fname in fnames:
		# 	fp = join(task_dir_path, fname)
		# 	print(f'Copying {fp}')
		# 	copyfile(fp, join(outDir, f'{idx}_{fname}'))
	else:
		print(f'ERROR (copy_logs_to_scratch): outDir ({outDir}) already exists')
