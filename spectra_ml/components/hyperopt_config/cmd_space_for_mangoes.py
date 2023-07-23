from math import ceil
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from pprint import pprint, pformat
import numpy as np
from hyperopt.pyll.base import scope
from scipy.stats import norm
import math

from spectra_ml.components.hyperopt_config.helpers import hp_ordinal_randint, hp_better_quniform, round_to_n, round_to_3, hp_idx, make_odd

__author__ = "Matthew Dirks"

NUM_TRAINING_RUNS = 40

# Hyperparameter search space variants specified here:
# (keys must correspond to values of `which_cmd_space`)
SPACE_VARIANTS = {
	'MANGOES_NN1': {
		'SENSORS': ['NIR_truncated'],
		'BATCH_SIZE_OPTIONS': [128],
		'BASE_LR_OPTIONS': [1e-2, 1e-3, 1e-4],
		'L2_RANGE': (0.0001, 100),

		'CONV_WIDTH_OPTIONS': [3, 5, 7, 11, 15, 21, 29, 41, 57, 79, 111, 155, 217],
		'CONV_N_FILTERS': (1, 5),

		'NUM_HIDDEN_LAYERS': 0,
	},
	'MANGOES_NN2': {
		'SENSORS': ['NIR_truncated'],
		'BATCH_SIZE_OPTIONS': [128],
		'BASE_LR_OPTIONS': [1e-2, 1e-3, 1e-4],
		'L2_RANGE': (0.0001, 100),

		'CONV_WIDTH_OPTIONS': [11],
		'CONV_N_FILTERS': (4, 4),

		'NUM_HIDDEN_LAYERS': 1,
		'FC_SIZE_RANGE': (1, 20),
	},
}


def get_cmd_space(which_cmd_space, target):
	cmd_space = _get_cmd_space(which_cmd_space, target)

	hyperhyperparams = {
		'WHICH_SET': 'dev',
		'WHICH_METRIC': 'RMSE',
	}

	return cmd_space, hyperhyperparams

def _get_cmd_space(which_cmd_space, target):
	""" 
	Returns a dictionary, named cmd_space, specifying the hyperparameter search space.
	I call it a "cmd_space" because these hyperparameters are passed to `train_TF_model.py`
	via command-line arguments.

	I use dollar signs ($) in some keys of cmd_space to remind me that these are 
	hyperparameters which are later converted via `_process_function` below. 
	When `_process_function` is done with them, these key:value pairs are removed from
	the dictionary because they are not command-line arguments.

	The keys that do not start with a "$" directly correspond to command-line arguments accepted
	by the `build_and_train` function in `train_TF_model.py`

	Hyperopt has its own internal set of names for each hyperparameter.
	By convention, I set this name to correspond to the key in cmd_space.
	"""
	sensor_type = SPACE_VARIANTS[which_cmd_space]['SENSORS'][0]
	assert len(SPACE_VARIANTS[which_cmd_space]['SENSORS']) == 1

	SPACE_VARIANT = SPACE_VARIANTS[which_cmd_space]

	def _process_function(cmd_dict):
		""" This function is called on EACH cmd_space sample (so, each "trial" from hyperopt)
		and it converts the hyperparameter values into command-line args that 
		`train_TF_model.py` can recognize, where needed. """

		# select which combo of conv widths to use
		idx = cmd_dict['$conv_filter_width_idx']
		conv_filter_width = cmd_dict[f'$CONV_WIDTH_OPTIONS'][idx]

		batch_size_idx = cmd_dict['$batch_size_idx']
		batch_size = cmd_dict['$BATCH_SIZE_OPTIONS'][batch_size_idx]

		base_LR_idx = cmd_dict['$base_LR_idx']
		base_LR = cmd_dict['$BASE_LR_OPTIONS'][base_LR_idx]

		### Save to cmd_dict
		cmd_dict.update({
			'base_LR': base_LR,
			'batch_size': batch_size,
			'conv_filter_width': conv_filter_width,
		})

		# if there's a hidden layer, select how many units to use for it
		if ('$FC_size' in cmd_dict):
			assert 'FC_size_per_input' not in cmd_dict
			FC_size = cmd_dict['$FC_size']
			
			# put the L2 reg factors into a list (one per sensor; in this case there's only 1)
			FC_L2_reg_factor_per_input = [cmd_dict['$FC_L2_reg_factor_0']]

			cmd_dict.update({
				'FC_size_per_input': [FC_size],
				'FC_L2_reg_factor_per_input': FC_L2_reg_factor_per_input,
			})
		else:
			assert 'FC_size_per_input' not in cmd_dict
			assert '$FC_L2_reg_factor_0' not in cmd_dict
		
		# remove the temporary key:values that were used in processing above (they start with "$")
		for key in list(cmd_dict.keys()):
			if (key.startswith('$')):
				del cmd_dict[key]
				
		return cmd_dict
	
	""" Note about arrays in hyperopt:
	Hyperopt does have a `choice` function for specifying hyperparameters whose values are selected
	from a list. For example:
		hp.choice('FC_init', ['xavier', 'he_normal'])
	but for hyperparameters where the values are ordered, like a list of kernel widths, say:
		[5, 10, 15, 20]
	hyperopt will treat this as unordered, and won't treat the performance of 5 as being more
	related to 10 than to 20.
	For this reason, we made a `hp_idx` function which creates an ordinal random integer variable
	that is treated as an index into a list of options.
	"""

	#### LISTS OF OPTIONS ####
	base = 1.4
	n = 15
	DEFAULT_CONV_WIDTH_OPTIONS = remove_dups([make_odd(base**i) for i in range(15)])
	DEFAULT_CONV_WIDTH_OPTIONS.remove(1) # don't do width=1, that's just silly
	# for n = 15: widths are [3, 5, 7, 11, 15, 21, 29, 41, 57, 79, 111]
	# for n = 24: 3,5,7,11,15,21,29,41,57,79,111,155,217,305,427,597,837,1171,1639,2295
	
	CONV_WIDTH_OPTIONS = SPACE_VARIANT.get('CONV_WIDTH_OPTIONS', DEFAULT_CONV_WIDTH_OPTIONS)
	BATCH_SIZE_OPTIONS = SPACE_VARIANT['BATCH_SIZE_OPTIONS']
	L2_RANGE = SPACE_VARIANT.get('L2_RANGE', (1e-4, 10)) # default when not specified is (1e-4, 10)
	BASE_LR_OPTIONS = SPACE_VARIANT.get('BASE_LR_OPTIONS', [1e-3, 1e-4, 1e-5, 1e-6])
	CONV_N_FILTERS = SPACE_VARIANT.get('CONV_N_FILTERS', (1, 5))
	PROX_L2_RANGE = SPACE_VARIANT.get('PROX_L2_RANGE')
	BN = SPACE_VARIANT.get('BN', False)

	options = {
		'$BASE_LR_OPTIONS': BASE_LR_OPTIONS, 
		'$BATCH_SIZE_OPTIONS': BATCH_SIZE_OPTIONS,
		'$CONV_WIDTH_OPTIONS': CONV_WIDTH_OPTIONS, 
	}

	cmd_space = {}
	cmd_space.update(options)
	cmd_space.update({
		'dataset_name': "mangoes",
		'input_features': SPACE_VARIANT['SENSORS'],
		'scaler_settings': {'X_type': 'mean_std', 'Y_type': 'mean_std'},
		'n_training_runs': NUM_TRAINING_RUNS,
		'fold_spec': {'type': '10fold_and_test_split', 'use_dev': True},
		'which_targets_set': target,
		'which_folds': list(range(10)),

		'n_full_epochs': 10000,
		'LR_sched_settings': {'type': 'ReduceLROnPlateau', 'patience': 25, 'factor': 0.5, 'base_min_LR': 1e-11},

		'$base_LR_idx': hp_idx('base_LR_idx', options['$BASE_LR_OPTIONS']),
		'$batch_size_idx': hp_idx('batch_size_idx', options['$BATCH_SIZE_OPTIONS']),
		'do_ES': True,
		'ES_patience': SPACE_VARIANT.get('ES_PATIENCE', 50),

		'conv_filter_init': 'he_normal',
		# 'conv_filter_width': 21,
		'$conv_filter_width_idx': hp_idx('conv_filter_width_idx', options[f'$CONV_WIDTH_OPTIONS']),
		'conv_L2_reg_factor': None,  # before 2023-04-05 this was 0
		'conv_proximity_L2_factor': None if (PROX_L2_RANGE is None) else hp.loguniform('conv_proximity_L2_factor', np.log(PROX_L2_RANGE[0]), np.log(PROX_L2_RANGE[1])),
		'conv_n_filters': hp_better_quniform('conv_n_filters', CONV_N_FILTERS[0], CONV_N_FILTERS[1], 1),

		'FC_L2_reg_factor': hp.loguniform('FC_L2_reg_factor', np.log(L2_RANGE[0]), np.log(L2_RANGE[1])),
		'FC_init': 'he_normal',

		'BN': BN,

		'$process_function': _process_function,
	})

	# Different experiments will use different number of hidden layers (and units) for the dense layers
	num_hidden_layers = SPACE_VARIANT['NUM_HIDDEN_LAYERS']
	if (num_hidden_layers == 0):
		assert 'FC_size_per_input' not in cmd_space
	elif (num_hidden_layers == 1):
		cmd_space.update({
			'$FC_size': hp_better_quniform('FC_size', *SPACE_VARIANT['FC_SIZE_RANGE'], 1),
			'$FC_L2_reg_factor_0': hp.loguniform('FC_L2_reg_factor_0', np.log(L2_RANGE[0]), np.log(L2_RANGE[1])),
		})
	else:
		raise(ValueError('num_hidden_layers > 1 not supported'))

	return cmd_space

def remove_dups(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]
