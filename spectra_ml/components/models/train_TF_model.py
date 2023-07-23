from datetime import datetime as dt
import os
from os.path import join, split, realpath, exists, isdir
import logging
from colorama import Fore, Back, Style
import colorama
from timeit import default_timer as timer
import uuid
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from time import sleep
import traceback
import sys
APP_DIR = split(realpath(__file__))[0]
sys.path.append(APP_DIR)

from spectra_ml.components.ensembling import Ensembler

try:
	from arca2.mypushbullet import notify
except:
	# if notify function unavilable, do nothing
	def notify(*_): pass

__author__ = "Matthew Dirks"

mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING)
pb_logger = logging.getLogger("pushbullet.Listener")
pb_logger.setLevel(logging.WARNING)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress some tensorflow messages
#from tensorflow.python.util import deprecation as tfdeprecation
#tfdeprecation._PRINT_DEPRECATION_WARNINGS = False

SETS = ['train', 'dev', 'test']
COMPUTER_NAME = socket.gethostname()

class PretendFuture:
	""" This pretends to be a "future" from the `concurrent` module (for when parallelization not in use) """
	def __init__(self, the_result):
		self.the_result = the_result
	def result(self):
		return self.the_result

def do_training_run(ith_run, out_dir, dataset_dict, hyperparams, configData, target_columns_in_use, kth_fold, fold_spec, scaler_settings, input_features, seed_start=None):
	""" This runs in a subprocess """

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

	# imports are here to make subprocesses work
	import tensorflow as tf
	# tf.get_logger().setLevel('ERROR')

	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu_instance in gpus: 
		tf.config.experimental.set_memory_growth(gpu_instance, True)

	LOG_TEXT = '[do_training_run]'
	LOG_TEXT += f'\n\tGPUs: {gpus}'

	if (seed_start is not None):
		seed = seed_start + ith_run
		LOG_TEXT += f'\n\tseed: {seed}'
	else:
		seed = None
		LOG_TEXT += '\n\tseed: None (not seeding; random will be used)'

	from spectra_ml.components.spectroscopy_NN.dynamic_model import reset_then_init_then_train
	run_result, model_cnn = reset_then_init_then_train(ith_run, seed, dataset_dict, hyperparams, configData, target_columns_in_use, kth_fold, fold_spec, scaler_settings, input_features)
	run_result['DEBUG'] = LOG_TEXT + '\n' + run_result['DEBUG']


	if (configData['NN_models']['save_checkpoints']):
		# run_result['model_cnn'] = model_cnn # doesnt work because "Can't pickle local object 'Layer.add_loss.<locals>._tag_callable'" when running parallel processes

		# import is here to make sure subprocess works
		from spectra_ml.components.spectroscopy_NN.end_of_training import get_run_out_dir
		out_dir_for_run = get_run_out_dir(ith_run, out_dir)
		if (not exists(out_dir_for_run)):
			os.makedirs(out_dir_for_run)

		# save the model (weights only)
		# model_cnn.save_weights(join(out_dir_for_run, 'saved_checkpoint'))

		# lets try saving the WHOLE model (architecture AND weights)
		model_cnn.save(join(out_dir_for_run, 'saved_model.h5'))

	return ith_run, run_result
	
def tf_check():
	""" Verify tensorflow version and that GPU is available. This runs in a subprocess because tensorflow behaves 
	poorly if imported in a main process AND in a child process, so I ONLY import it in child processes
	"""
	print('\n' + '='*100 + '\n', flush=True)
	try:
		import tensorflow as tf
		assert tf.__version__ == '2.6.0'
	except:
		raise(ImportError('tensorflow not found; be sure to use correct environment (conda activate tf2)'))

	gpus = tf.config.experimental.list_physical_devices('GPU')
	print('[tf_check] GPUs: ', gpus, flush=True)

	print('\n' + '='*100 + '\n', flush=True)
	return len(gpus)

def train_TF_model(
	#=== THESE COME FROM `train_model.py` ===
		configData=None,
		out_dir=None,
		start_datetime=None,
		dataset_dict=None,
		target_columns_in_use=None,

	#=== THESE COME DIRECTLY FROM COMMAND LINE (OR OVERRIDDEN BY CMD_ARGS_FPATH) === 
		input_features=['XRF'],
		fold_spec=None,
		scaler_settings=None,
		kth_fold=0,
		n_gpu=1,
		n_in_parallel=None,
		seed_start=None,
		LR_finder_settings=None,

		# optimizer and training hyperparams:
		n_training_runs=None, # for ensembling
		base_LR=None,
		n_full_epochs=None,
		batch_size=None,
		do_ES=True,
		ES_patience=None, # TF default is 50
		LR_sched_settings=None,
		epsilon=None, # TF default is 1e-7

		# hyperparams (for architecture)
		conv_filter_init=None,
		conv_filter_width=None,
		conv_L2_reg_factor=None,
		conv_n_filters=None,
		FC_init=None,
		FC_L2_reg_factor=None,
		FC_size_per_input=None,
		FC_L2_reg_factor_per_input=None,
		conv_proximity_L2_factor=None,

		BN=False, # whether to use BatchNormalization
	):
	logger = logging.getLogger('spectra_ml')

	# imports are here to make subprocesses work
	from spectra_ml.components.spectroscopy_NN.end_of_training import finish_training_run, finish_ensemble

	##################### CHECK/FIX ARGUMENTS #################################
	assert n_in_parallel is not None, 'Please specify'
	assert n_training_runs is not None, 'Please specify'
	assert kth_fold is not None, 'Please specify'
	assert fold_spec is not None, 'Please specify'
	assert scaler_settings is not None, 'Please specify'

	# bash sometimes pases 0-padded numbers which get intrepreted as a string, rather than int (e.g. "01")
	if (seed_start is not None):
		seed_start = int(seed_start)

	if (n_in_parallel > 1): # if multiple processes will be running at once...
		assert not configData['plot']['conv_filters'] and not configData['plot']['FC_weights'], 'plot happens within the subprocess, but mpl not thread safe (sometimes it places a figure from one thread into the figure of another thread). Either turn off plotting or run serially.'

	##################### HYPERPARAMETERS FOR NN TRAINING #######################
	# these get passed-through to each "worker" that will each do one training run
	hyperparams = {
		'base_LR': base_LR,
		'n_epochs': n_full_epochs,

		'conv_filter_init': conv_filter_init,
		'conv_filter_width': conv_filter_width,
		'conv_n_filters': conv_n_filters,
		'FC_init': FC_init,

		'do_ES': do_ES, # if False, just run training through to the last `n_epochs` without early stopping
		'LR_sched_settings': LR_sched_settings,
		'BN': BN,
	}

	for key, value in hyperparams.items():
		assert value is not None, f'Please specify {key} argument'


	# THESE ARE OPTIONAL HYPERPARAMS THAT CAN BE 'None'
	hyperparams.update({
		'batch_size': batch_size,
		'LR_finder_settings': LR_finder_settings, # for dev/testing only
		'FC_size_per_input': FC_size_per_input,
		'FC_L2_reg_factor': FC_L2_reg_factor,
		'FC_L2_reg_factor_per_input': FC_L2_reg_factor_per_input, # one per input_feature (e.g. one per sensor)
		'ES_patience': ES_patience,
		'epsilon': epsilon,
		'conv_L2_reg_factor': conv_L2_reg_factor,
		'conv_proximity_L2_factor': conv_proximity_L2_factor,
	})

	if (do_ES):
		assert ES_patience is not None


	############################# TENSORFLOW CHECK ###############################
	logger.info('\n===TENSORFLOW CHECK===')

	# I've seen this set to 0 even when GPU is available (e.g. a Tesla V100 when I tested)
	logger.info(f'CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")}')

	# this attempts to ensure that a GPU won't be used, even if available, when user specifies n_gpu=0
	if (n_gpu == 0):
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	# time it
	d0 = dt.now()

	# run TensorFlow and GPU check in separate process
	n_available_gpus = 0
	ith_check = 0
	while True:
		logger.info(f'Doing TF and GPU check ({ith_check})')

		tf_checker = ProcessPoolExecutor(1).submit(tf_check)
		try:
			n_available_gpus = tf_checker.result(80)
		except TimeoutError as e:
			logger.error(f'TimeoutError caught ({ith_check})')
			logger.error(traceback.format_exc())
			# notify('PYTHON train_TF_model.py', f'TimeoutError {e}. out_dir={out_dir} @{COMPUTER_NAME}')
		except Exception as e:
			logger.error(f'Exception during tf_check ({ith_check}): {e}')
			logger.error(traceback.format_exc())
			notify('PYTHON train_TF_model.py', f'Exception during tf_check: {traceback.format_exc()}. out_dir={out_dir} @{COMPUTER_NAME}')
		else:
			break

		ith_check += 1
		if (ith_check >= 5):
			logger.critical(f'tf_checker keeps failing (has failed {ith_check} times). Giving up and exiting now')
			notify('PYTHON train_TF_model.py', f'tf_check failed. out_dir={out_dir} @{COMPUTER_NAME}')
			exit(1)

	if (n_gpu > 0):
		assert n_gpu <= n_available_gpus

	# print time taken
	d1 = dt.now()
	logger.info(f'Time taken to do tf_check: {d1 - d0} (hh:mm:ss.micro)')


	########################################### RUN TRAINING (IN PARALLEL IF REQUESTED) ##################################################
	args_for_function = (out_dir, dataset_dict, hyperparams, configData, target_columns_in_use, kth_fold, fold_spec, scaler_settings, input_features, seed_start)

	# PARALLELIZATION - n_in_parallel:
	as_completed_FUNCTION = as_completed
	futures = [] # results stored here
	#   - 0 means sequential, no process pool. 
	#   - setting to 1 is also sequential but still uses the subprocess function (for testing subprocess functionality)
	if (n_in_parallel == 0): 
		as_completed_FUNCTION = lambda _futures: _futures # override `as_completed` from `concurrent.futures`

		logger.info('\n===RUNNING SEQUENTIALLY WITHOUT SUBPROCESS===')
		for ith_run in range(n_training_runs):
			print(f'Running {ith_run}')
			result = do_training_run(ith_run, *args_for_function)
			futures.append(PretendFuture(the_result=result)) # this object pretends to be a "future" like from the ProcessPoolExecutor

	else:
		# === Schedule training runs in pool
		logger.info('\n===MAKING PROCESS POOL===')
		executor = ProcessPoolExecutor(n_in_parallel)
		for ith_run in range(n_training_runs):
			""" methods of a `future`:
				  future.done(): True or False
				  future.result(): waits for it to complete then returns result
			"""
			# if there are any run-specific settings, deepcopy will be required otherwise jobs will share the same reference(s) to the variables
			future = executor.submit(do_training_run, ith_run, *args_for_function)
			print(f'Submitted {ith_run}')
			futures.append(future)

	# === Wait for results to come in, and process them in the order they happen to finish
	ensembler = Ensembler()
	for future in as_completed_FUNCTION(futures):
		ith_run, run_result = future.result()
		logger.info(f'=== TRAINING RUN {ith_run} DONE ===')

		finish_training_run(out_dir, configData, ith_run, run_result, target_columns_in_use)

		# add results to ensembler
		ensembler.record_metadata_for_run(ith_run, run_result)

	logger.info(f'\n\n===ALL TRAINING RUNS COMPLETED=== (started at {start_datetime}, now is {dt.now()})')

	predictions_dict = finish_ensemble(out_dir, configData, ensembler, target_columns_in_use, kth_fold)
	logger.info(f'\n=== finish_ensemble DONE === (started at {start_datetime}, now is {dt.now()})')

	return predictions_dict

# if __name__ == '__main__':
# 	colorama.init(wrap=False)
#	from spectra_ml.components.cmd_line_helper import get_main_function
# 	fire.Fire(get_main_function(train_TF_model))