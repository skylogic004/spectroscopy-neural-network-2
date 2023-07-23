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
	'MANGOES_NN3': { # Sensor Fusion model (treating the truncated NIR spectrum and preprocessed smoothed spectrum (SNV_SG1) like "sensors" or "blocks")
		'SENSORS': ['NIR_truncated', 'SNV_SG1'], # NOTE! each sensor type has hardcoded settings in this file.
		'BATCH_SIZE_OPTIONS': [128],
		'BASE_LR_OPTIONS': [1e-4],

		'FC_L2_REG_FACTOR': {'mu': 1.8035777327468836, 'sigma': 0.8443541186257424},

		'CONV_FILTER_WIDTH': {'NIR_truncated': 11, 'SNV_SG1': 9},
		'CONV_N_FILTERS': {'NIR_truncated': 1, 'SNV_SG1': 1},

		# for the hidden FC layer:
		'NUM_HIDDEN_LAYERS': 1,
		'FC_SIZES': {
			'NIR_truncated': 3, 
			'SNV_SG1': 3
		},
		'FC_L2_PER_SENSOR': {
			'NIR_truncated': {'mu': -3.7900874036213197, 'sigma': 1.151292546497023}, 
			'SNV_SG1': {'mu': -6.993689054974152, 'sigma': 1.151292546497023}, 
		},
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

	SPACE_VARIANT = SPACE_VARIANTS[which_cmd_space]

	def _process_function(cmd_dict):
		""" This function is called on EACH cmd_space sample (so, each "trial" from hyperopt)
		and it converts the hyperparameter values into command-line args that 
		`train_TF_model.py` can recognize, where needed. """

		batch_size_idx = cmd_dict['$batch_size_idx']
		batch_size = cmd_dict['$BATCH_SIZE_OPTIONS'][batch_size_idx]

		base_LR_idx = cmd_dict['$base_LR_idx']
		base_LR = cmd_dict['$BASE_LR_OPTIONS'][base_LR_idx]

		### Save to cmd_dict
		cmd_dict.update({
			'base_LR': base_LR,
			'batch_size': batch_size,
		})

		# if there's a hidden layer, select how many units to use for it
		if ('FC_size_per_input' in cmd_dict):
			# put the L2 reg factors into a list (one per sensor)
			cmd_dict.update({
				'FC_L2_reg_factor_per_input': [
					# cmd_dict['$FC_L2_reg_factor_XRF'], 
					# cmd_dict['$FC_L2_reg_factor_HYPER'], 
					# cmd_dict['$FC_L2_reg_factor_LIBS'],
					cmd_dict['$FC_L2_reg_factor_NIR_truncated'],
					cmd_dict['$FC_L2_reg_factor_SNV_SG1'],
				],
			})
		else:
			# assert '$FC_L2_reg_factor_XRF' not in cmd_dict
			# assert '$FC_L2_reg_factor_HYPER' not in cmd_dict
			# assert '$FC_L2_reg_factor_LIBS' not in cmd_dict
			assert '$FC_L2_reg_factor_NIR_truncated' not in cmd_dict
			assert '$FC_L2_reg_factor_SNV_SG1' not in cmd_dict
		
		# remove the temporary key:values that were used in processing above (they start with "$")
		for key in list(cmd_dict.keys()):
			if (key.startswith('$')):
				del cmd_dict[key]
				
		return cmd_dict
	

	d = SPACE_VARIANT['CONV_FILTER_WIDTH']
	# conv_filter_width = [d['XRF'], d['HYPER'], d['LIBS']]
	conv_filter_width = [d['NIR_truncated'], d['SNV_SG1']]

	d = SPACE_VARIANT['CONV_N_FILTERS']
	# conv_n_filters = [d['XRF'], d['HYPER'], d['LIBS']]
	conv_n_filters = [d['NIR_truncated'], d['SNV_SG1']]

	BN = SPACE_VARIANT.get('BN', False)

	options = {
		'$BASE_LR_OPTIONS': SPACE_VARIANT['BASE_LR_OPTIONS'], 
		'$BATCH_SIZE_OPTIONS': SPACE_VARIANT['BATCH_SIZE_OPTIONS'],
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
		'conv_L2_reg_factor': 0, 
		'conv_filter_width': conv_filter_width,
		'conv_n_filters': conv_n_filters,

		'FC_init': 'he_normal',

		'BN': BN,

		# L2 for the common branch
		'FC_L2_reg_factor': hp.lognormal('FC_L2_reg_factor', SPACE_VARIANT['FC_L2_REG_FACTOR']['mu'], SPACE_VARIANT['FC_L2_REG_FACTOR']['sigma']),

		'$process_function': _process_function,
	})

	# Different experiments will use different number of hidden layers (and units) for the dense layers
	num_hidden_layers = SPACE_VARIANT['NUM_HIDDEN_LAYERS']
	if (num_hidden_layers == 0):
		assert 'FC_SIZES' not in SPACE_VARIANT
		assert 'FC_L2_PER_SENSOR' not in SPACE_VARIANT
		assert 'FC_size_per_input' not in cmd_space
	elif (num_hidden_layers == 1):
		assert 'FC_SIZES' in SPACE_VARIANT
		assert 'FC_L2_PER_SENSOR' in SPACE_VARIANT

		cmd_space.update({
			# '$FC_size': hp_better_quniform('FC_size', *SPACE_VARIANT['FC_SIZE_RANGE'], 1),
			'FC_size_per_input': [
				SPACE_VARIANT['FC_SIZES']['NIR_truncated'],
				SPACE_VARIANT['FC_SIZES']['SNV_SG1'],
			],

			# L2 for each of the 3 sensors (in addition to the one for the common branch):
			'$FC_L2_reg_factor_NIR_truncated': hp.lognormal('FC_L2_reg_factor_NIR_truncated', SPACE_VARIANT['FC_L2_PER_SENSOR']['NIR_truncated']['mu'], SPACE_VARIANT['FC_L2_PER_SENSOR']['NIR_truncated']['sigma']),
			'$FC_L2_reg_factor_SNV_SG1': hp.lognormal('FC_L2_reg_factor_SNV_SG1', SPACE_VARIANT['FC_L2_PER_SENSOR']['SNV_SG1']['mu'], SPACE_VARIANT['FC_L2_PER_SENSOR']['SNV_SG1']['sigma']),
			# '$FC_L2_reg_factor_LIBS': hp.lognormal('FC_L2_reg_factor_LIBS', SPACE_VARIANT['FC_L2_PER_SENSOR']['LIBS']['mu'], SPACE_VARIANT['FC_L2_PER_SENSOR']['LIBS']['sigma']),
		})
	else:
		raise(ValueError('num_hidden_layers > 1 not supported'))

	return cmd_space
