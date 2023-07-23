"""
This python module allows creating a custom NN architecture (dynamically).
Code follows a similar structure to `baseline_model.py`.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import random
import logging
from timeit import default_timer as timer
import datetime

from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input, Conv1D, Reshape, concatenate, BatchNormalization, Activation
from keras.models import Model
import pandas as pd
from tensorflow.keras.regularizers import L2

from spectra_ml.components.data_loader.data_structure import CurrentData
from spectra_ml.components.plot.conv_filters import plot_conv_filters
from spectra_ml.components.plot.FC_weights import plot_FC_weights

__author__ = "Matthew Dirks"

rmse = lambda arr1, arr2: np.sqrt(mean_squared_error(arr1, arr2))
is_list_or_tuple = lambda value: isinstance(value, list) or isinstance(value, tuple)

SETS = ['train', 'dev', 'test']

CAP_NEGATIVE_PREDICTIONS = True
# Setting to True assumes that negative predictions are NOT plausible in this domain!

class SimpleLogger:
	def __init__(self, ith_run):
		self.log_txt = ''
		self.ith_run = ith_run
	def log(self, new_msg):
		self.log_txt += new_msg + '\n'
	def log_on_first_run(self, new_msg):
		if (self.ith_run == 0):
			self.log_txt += new_msg + '\n'

def set_all_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	random.seed(seed)
	tf.random.set_seed(seed)

def get_init_fn(init_name, seed=None):
	init_name = init_name.lower()

	if (init_name == 'unit_normal'):
		return tf.random_normal_initializer(stddev=1, seed=seed)

	elif (init_name.startswith('xavier')): 
		# xavier is another name for glorot 
		#   https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal
		#   https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
		raise(ValueError('Use glorot instead of xavier'))

	elif (init_name == 'zero'):
		return tf.initializers.constant(0)

	elif (init_name == 'glorot_normal'):
		return tf.keras.initializers.glorot_normal(seed=seed)

	elif (init_name == 'glorot_uniform'):
		return tf.keras.initializers.glorot_uniform(seed=seed)

	elif (init_name == 'he_normal'):
		return tf.keras.initializers.he_normal(seed=seed)

	elif (init_name == 'he_uniform'):
		return tf.keras.initializers.he_uniform(seed=seed)

	else:
		raise(ValueError('Invalid initialization function name: {}'.format(init_name)))

class ProxL2(tf.keras.regularizers.Regularizer):
	""" Based on structure of Keras K2 regularizer here: https://github.com/keras-team/keras/blob/v2.11.0/keras/regularizers.py#L291-L323

	NOTE: not compatible with saving checkpoints. It throws this Exception:
		NotImplementedError: <spectra_ml.components.spectroscopy_NN.dynamic_model.ProxL2 object at 0x0000013CBB758640> does not implement get_config()
	"""

	def __init__(self, factor, conv_width):
		self.factor = tf.keras.backend.cast_to_floatx(factor)
		self.factor_scaled = self.factor * conv_width

	def __call__(self, x):
		# shape is [width, 1, num_filters]
		# e.g. shape=(111, 1, 3)
		_diff_to_neighbors = x[1:, :, :] - x[:-1, :, :]
		
		# The following 2 formulas are equivalent except the latter avoids the extra division operation on every call
		# loss = self.factor * tf.reduce_mean(_diff_to_neighbors**2, name='mean_squared_proximity_diff')
		loss = self.factor_scaled * tf.reduce_sum(_diff_to_neighbors**2, name='mean_squared_proximity_diff')
		
		return loss


def init_model(hyperparams, input_info, seed=None):
	def get_seed(add):
		if (seed is not None):
			return seed + add
		else:
			return None

	# input features (may be different sensors or different pre-processed spectra) or "blocks"
	n_inputs = len(input_info)

	# if conv filter width is an int, it will be used for all conv layers,
	# if it's a tuple, there must be one width per input feature
	conv_filter_width = hyperparams['conv_filter_width']
	if (isinstance(conv_filter_width, int)):
		width_per_input = [conv_filter_width] * n_inputs
	elif (is_list_or_tuple(conv_filter_width)):
		width_per_input = conv_filter_width
	else:
		raise(ValueError(f'Invalid conv_filter_width ({type(conv_filter_width)})'))

	conv_n_filters = hyperparams['conv_n_filters']
	if (isinstance(conv_n_filters, int)):
		n_filters_per_input = [conv_n_filters] * n_inputs
	elif (is_list_or_tuple(conv_n_filters)):
		n_filters_per_input = conv_n_filters
	else:
		raise(ValueError(f'Invalid conv_n_filters ({type(conv_n_filters)})'))

	assert len(n_filters_per_input) == len(width_per_input) == n_inputs

	conv_filter_init = hyperparams['conv_filter_init']


	FC_init = hyperparams['FC_init']
	# Use FC_size_per_input for ONE hidden layer. Leave it out for ZERO hidden layers.
	FC_size_per_input = hyperparams['FC_size_per_input']
	FC_L2_reg_factor_per_input = hyperparams['FC_L2_reg_factor_per_input']

	branch_models = []
	for input_idx, (input_info, tmp_conv_width, tmp_n_filters) in enumerate(zip(input_info, width_per_input, n_filters_per_input)):
		size = input_info['size']
		feature = input_info['feature']

		the_input = Input(shape=(size,), name=f'input_{feature}')

		prev_layer = Reshape((size, 1), input_shape=(size,))(the_input)

		# regularizer for conv layer, if any
		conv_L2_reg_factor = hyperparams['conv_L2_reg_factor']
		if (conv_L2_reg_factor is not None):
			assert 'conv_proximity_L2_factor' not in hyperparams or hyperparams['conv_proximity_L2_factor'] is None, f'it\'s set to {conv_proximity_L2_factor}'
			conv_kernel_regularizer = L2(conv_L2_reg_factor)
		elif (hyperparams.get('conv_proximity_L2_factor') is not None):
			assert hyperparams['conv_proximity_L2_factor'] != 0
			conv_kernel_regularizer = ProxL2(hyperparams['conv_proximity_L2_factor'], tmp_conv_width)
		else:
			conv_kernel_regularizer = None

		prev_layer = Conv1D(filters=tmp_n_filters, 
							kernel_size=tmp_conv_width, 
							strides=1, 
							padding=hyperparams.get('conv_padding', 'same'),  # defaults to 'same' (pad with 0s)
							kernel_initializer=get_init_fn(conv_filter_init, get_seed(input_idx)),
							kernel_regularizer=conv_kernel_regularizer,
							activation=None, # NEW: USING BATCH. 'elu',
							input_shape=(size,1),
							name=f'conv1d_{feature}')(prev_layer)
		
		prev_layer = Flatten(name=f'flatten_{feature}')(prev_layer)

		if (hyperparams['BN']):
			prev_layer = BatchNormalization()(prev_layer)

		prev_layer = Activation('elu')(prev_layer)

		if (FC_size_per_input is not None):
			# Should be 1 per input
			assert len(FC_size_per_input) == n_inputs
			tmp_size = FC_size_per_input[input_idx]

			# Should be 1 FC reg factor per input
			assert FC_L2_reg_factor_per_input is not None
			assert len(FC_L2_reg_factor_per_input) == n_inputs
			tmp_L2_factor = FC_L2_reg_factor_per_input[input_idx]

			# for dense_idx, tmp_size in enumerate(FC_layer_sizes):
			dense_idx = 0
			prev_layer = Dense(tmp_size, 
							   kernel_initializer=get_init_fn(FC_init, get_seed((input_idx*100)+dense_idx)), 
							   kernel_regularizer=L2(tmp_L2_factor), 
							   activation=None, 
							   # activation='elu',
							   name=f'dense_{feature}_{dense_idx}')(prev_layer)

			if (hyperparams['BN']):
				prev_layer = BatchNormalization()(prev_layer)

			prev_layer = Activation('elu')(prev_layer)


		# END THIS BRANCH
		branch_models.append(Model(inputs=the_input, outputs=prev_layer))

	# COMBINE BRANCHES INTO ONE
	combined = concatenate([branch_model.output for branch_model in branch_models])

	last_layer = Dense(1, 
					   kernel_initializer=get_init_fn(FC_init, get_seed(1000)), 
					   kernel_regularizer=L2(hyperparams.get('FC_L2_reg_factor', None)), 
					   activation='linear')(combined)

	model_cnn = Model(inputs=[branch_model.input for branch_model in branch_models], outputs=last_layer)

	return(model_cnn)

def reset_then_init_then_train(ith_run, seed, dataset_dict, hyperparams, configData, target_columns_in_use, kth_fold, fold_spec, scaler_settings, input_features):
	import tensorflow as tf
	logger = SimpleLogger(ith_run)
	start_datetime = datetime.datetime.now() # for printing total time in nice human-readable format

	####################################################

	# === Prepare the data
	if ('seed' in fold_spec):
		fold_spec = fold_spec.copy()
		fold_spec['seed'] = fold_spec['seed'] + ith_run # ensure seed is unique between runs

	cur_data = CurrentData(kth_fold, 
						   dataset_dict,
						   fold_spec,
						   input_features,
						   scaler_settings, 
						   combine_train_dev=False,
						   log_fn=logger.log)

	X_data_dict = cur_data.get_prepared_X()

	X = {}
	for which_set in cur_data.sets:
		X[which_set] = [X_data_dict[(which_set, feature)] for feature in input_features]

	num_training_examples = X['train'][0].shape[0]

	# === Seed
	tf.keras.backend.clear_session()
	if (seed is not None):
		set_all_seeds(seed)

	########### DEFINE HYPERPARAMETERS AND INSTANTIATE THE MODEL ###################
	input_info = [{'size': X_data_dict[('train', feature)].shape[1], 'feature': feature} for feature in input_features]
	model_cnn = init_model(hyperparams, input_info, seed)

	# print model summary (only on first run)
	model_cnn.summary(print_fn=logger.log_on_first_run)

	n_epochs = hyperparams['n_epochs']
	batch_size = hyperparams['batch_size']

	if (batch_size is None):
		# batch_size unspecified, so use the full size of the training set
		batch_size = num_training_examples


	base_LR_to_actual = lambda _base_LR: _base_LR * batch_size
	LR = base_LR_to_actual(hyperparams['base_LR'])

	########### COMPILE MODEL WITH ADAM OPTIMIZER #####################################
	model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, epsilon=hyperparams.get('epsilon', 1e-7)), loss='mse', metrics=['mse'])  

	callbacks = []

	########### LR RANGE FINDER###################################################
	LR_finder_settings = hyperparams['LR_finder_settings']
	if (LR_finder_settings is not None):
		assert hyperparams['LR_sched_settings'] == 'off'
		assert hyperparams['do_ES'] == False

		logger.log('LR schedule: disabled because doing LR finder.')
		from arca2.programs.spectra_ai.TF2.LRfinder import LRFinder
		max_steps = hyperparams['LR_finder_settings']['max_steps']
		lr_finder = LRFinder(max_steps=max_steps, batch_size=batch_size)
		callbacks.append(lr_finder)

		assert hyperparams['n_epochs'] == 0, 'Please set n_epochs (n_full_epochs) to 0 when using LR_finder because it is set automatically'
		n_epochs = int(np.ceil(max_steps * batch_size / num_training_examples))
		logger.log(f'n_epochs auto-set to {n_epochs}')

	########### LR SCHDULE #######################################################
	LR_sched_settings = hyperparams['LR_sched_settings']
	assert isinstance(LR_sched_settings, dict) or LR_sched_settings == 'off'

	if (LR_sched_settings == 'off'):
		logger.log_on_first_run('LR schedule: off')
	elif (LR_sched_settings['type'] == 'ReduceLROnPlateau'):
		assert all([key in ['type', 'patience', 'factor', 'base_min_LR'] for key in LR_sched_settings.keys()]), 'Invalid LR_sched_settings'

		if ('base_min_LR' in LR_sched_settings):
			min_LR = base_LR_to_actual(LR_sched_settings['base_min_LR'])
		else:
			min_LR = 1e-6 # the default

		scheduler_args = {
			'patience': LR_sched_settings.get('patience', 25),
			'factor': LR_sched_settings.get('factor', 0.5),
			'min_lr': min_LR,
		}
		logger.log_on_first_run(f'LR schedule: ReduceLROnPlateau. Parameters = {scheduler_args}')
		rdlr = ReduceLROnPlateau(**scheduler_args, monitor='val_loss', verbose=0)
		callbacks.append(rdlr)
	elif (LR_sched_settings['type'] == 'drop_LR_at_epochs'):
		logger.log_on_first_run('LR schedule: Custom callback, drops at epochs specified.')

		drop_LR_at_epochs = LR_sched_settings['drop_LR_at_epochs']

		assert isinstance(drop_LR_at_epochs, list)
		assert len(drop_LR_at_epochs) > 0
		

		def scheduler(epoch, lr):
			""" drop LR in half at the epochs specified in `drop_LR_at_epochs` list """
			#print(epoch, lr, epoch in drop_LR_at_epochs)
			if epoch in drop_LR_at_epochs:
				return lr/2
			else:
				return lr
		LR_sched_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
		callbacks.append(LR_sched_callback)

	else:
		raise(ValueError('Invalid LR_sched_settings'))


	########### EARLY STOPPING ######################################################
	if (hyperparams['do_ES']):
		early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=hyperparams['ES_patience'], mode='auto', restore_best_weights=True)
		callbacks.append(early_stop)

	########### TRAIN THE MODEL #####################################################
	h = model_cnn.fit(X['train'], 
					  cur_data.get_prepared_Y('train', target_columns_in_use), 
					  batch_size=batch_size, 
					  epochs=n_epochs,
					  validation_data=None if (X['dev'] is None) else (X['dev'], cur_data.get_prepared_Y('dev', target_columns_in_use)),
					  callbacks=callbacks, 
					  verbose=0)
	
	### RETURN RESULTS
	run_result = {
		'scores': {},
	}

	# save predictions
	# TODO: CODE DE-DUPLICATION. THIS CODE IS REPEATED (almost identical with some differences) TO: 
	#       dynamic_model.py, train_ROSA_model.py, train_sklearn_model.py
	predictions_dict = {}
	for which_set in cur_data.sets:
		# get groundtruth
		groundtruth_df = cur_data.get(which_set)[target_columns_in_use].copy()
		assert groundtruth_df.index.name == 'sampleId'
		groundtruth_df['set'] = which_set

		# make predictions (these will be normalized, if normalization is in use)
		predictions_normalized = model_cnn.predict(X[which_set])
		# numpy array with shape (n, d) (n = number of examples, d = number of targets)

		# Convert numpy array into DataFrame.
		# Note: column names must match name used in `normalize_y_data.py` for unnormalization function to work
		predictions_normalized_df = pd.DataFrame(predictions_normalized, columns=target_columns_in_use, index=groundtruth_df.index)

		# Unnormalize the data, using settings stored in cur_data
		predictions_unnormalized_df = cur_data.unnormalize_Y_df(predictions_normalized_df)

		# Cap negative predictions to 0:
		if (CAP_NEGATIVE_PREDICTIONS):
			predictions_unnormalized_df[predictions_unnormalized_df<0] = 0

		# calc RMSE (across all targets and all examples)
		try:
			score = rmse(groundtruth_df[target_columns_in_use], predictions_unnormalized_df)
		except ValueError:
			score = np.inf
			logger.log(f'RMSE for {which_set} set is set to inf because of ValueError')
		run_result['scores'][f'RMSE_{which_set}'] = score

		# rename columns from "Target" to "Target_pred" in the predictions DataFrame
		predictions_unnormalized_df.rename(columns={name:f'{name}_pred' for name in predictions_unnormalized_df.columns}, inplace=True)

		# save predictions and groundtruth
		predictions_dict[which_set] = groundtruth_df.join(predictions_unnormalized_df)

	run_result['seed'] = seed
	run_result['h.history'] = h.history
	run_result['predictions_dict'] = predictions_dict
	run_result['DEBUG'] = logger.log_txt
	run_result['time_spent'] = datetime.datetime.now() - start_datetime
	
	if (hyperparams['LR_finder_settings'] is not None):
		# run_result['LR_finder_fig'] = lr_finder.plot()
		fig, info = lr_finder.plot2()
		run_result['LR_finder_fig_2'] = fig
		run_result['LR_finder_info'] = info
		# import matplotlib.pyplot as plt
		# plt.show()


	if (configData['plot']['conv_filters']):
		run_result['conv_filters_fig'] = plot_conv_filters(model_cnn)

	if (configData['plot']['FC_weights']):
		run_result['FC_weights_fig'] = plot_FC_weights(model_cnn)

	return run_result, model_cnn

