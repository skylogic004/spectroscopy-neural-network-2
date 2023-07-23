import logging
import pandas as pd
import numpy as np
from collections import OrderedDict

def scale_x_via_max(scaler_settings,
					 X_data_dict, 
					 sets_in_use, 
					 input_features, 
					 importance_scales):
	""" Scales each input_feature by dividing by max. 
		Does NOT move bottom to 0 because we assume that spectra start pretty close to 0 anyways.
	"""
	logger = logging.getLogger('spectra')

	# for overriding the ones that would otherwise be computed in this function
	# (stats used to be called "input_maxes")
	X_stats = scaler_settings.get('X_stats')

	# get normalizing constant for all spectra, 1 per sensor type. training (& unsuper) set only
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			_counts_max = X_data_dict[('train', feature)].max()

			# also consider unsupervised data too, if available
			if (('unsuper', feature) in X_data_dict):
				_counts_max = max(_counts_max, X_data_dict[('unsuper', feature)].max())

			# not normalizing within model anymore. It screws with the loss function.
			# _max_tf = tf.constant(_counts_max, dtype=tf.float32)
			
			X_stats[feature] = {'max': _counts_max}
			logger.info(f'Input data for feature {feature} normalized via max on train set.') #: {X_stats[feature]}')
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized via max by user-provided weights (not computed here).') #:\n\t{}'.format(X_stats))

	# apply normalization to all matrices
	for feature in input_features:
		for _set in sets_in_use:
			X_data_dict[(_set, feature)] /= X_stats[feature]['max']

	# apply scaling to adjust a feature to be more "important" than another
	assert len(input_features)==len(importance_scales)
	for feature, scale in zip(input_features, importance_scales):
		for _set in sets_in_use:
			X_data_dict[(_set, feature)] *= scale

	return X_data_dict, scaler_settings

def scale_x_via_min_max(scaler_settings,
						 X_data_dict, 
						 sets_in_use, 
						 input_features):
	""" This version does normalization using min and max. Per input feature.
	"""
	logger = logging.getLogger('spectra')
	X_stats = scaler_settings.get('X_stats')


	# get normalizing constants for all spectra, 1 per sensor type. train set only.
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			X_stats[feature] = {
				'min': X_data_dict[('train', feature)].min(),
				'max':  X_data_dict[('train', feature)].max(),
			}
			logger.info(f'Input data for feature {feature} normalized via min_max on train set.') #: {X_stats[feature]}')
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized via min_max by user-provided weights (not computed here).') #:\n\t{}'.format(X_stats))

	# apply normalization to all matrices (all sets)
	for feature in input_features:
		for _set in sets_in_use:
			_min = X_stats[feature]['min']
			_max = X_stats[feature]['max']

			X_data_dict[(_set, feature)] -= _min
			X_data_dict[(_set, feature)] /= _max - _min

	return X_data_dict, scaler_settings

def scale_x_via_mean_std(scaler_settings,
						  X_data_dict, 
						  sets_in_use, 
						  input_features):
	""" This version does standardization using mean and std. Column-wise. """
	logger = logging.getLogger('spectra')
	X_stats = scaler_settings.get('X_stats')

	# get stats per "column" (i.e. each band of each spectrum, where a spectrum is an "input_feature")
	# on train set only
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			# calc mean - to subtract it
			means = X_data_dict[('train', feature)].mean(axis=0)

			# calc std - to divide by it
			stds = X_data_dict[('train', feature)].std(axis=0)
			num_zero_values = (stds==0).sum()
			if (num_zero_values > 0):
				logger.info(f'In calculating std for mean_std standardization, {num_zero_values} 0 values were found. Will use 1 instead of 0 in the division.')
				stds[stds==0] = 1

			X_stats[feature] = {
				'means': means,
				'stds': stds,
			}

			logger.info(f'Input data for feature {feature} normalized via mean_std on train set.') #: {X_stats[feature]}')
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized via mean_std by user-provided weights (not computed here).') #:\n\t{}'.format(X_stats))

	# apply normalization to all matrices (all sets)
	for feature in input_features:
		for _set in sets_in_use:
			means = X_stats[feature]['means']
			stds = X_stats[feature]['stds']

			X_data_dict[(_set, feature)] -= means
			X_data_dict[(_set, feature)] /= stds

	return X_data_dict, scaler_settings

def scale_x_via_PowerTransformer(scaler_settings,
						          X_data_dict, 
						          sets_in_use, 
						          input_features):
	""" This version does standardization using
	https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
	Column-wise. """
	from sklearn.preprocessing import PowerTransformer

	logger = logging.getLogger('spectra')
	X_stats = scaler_settings.get('X_stats')

	# get stats per "column" (i.e. each band of each spectrum, where a spectrum is an "input_feature")
	# on train set only
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			pt = PowerTransformer()
			pt.fit(X_data_dict[('train', feature)])

			X_stats[feature] = {
				'power_transformer': pt,
				'lambdas': pt.lambdas_,
			}
			logger.info(f'Input data for feature {feature} normalized via PowerTransformer on train set.') #: {X_stats[feature]}')
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized via PowerTransformer by user-provided weights (not computed here).') #:\n\t{}'.format(X_stats))
		raise(ValueError('This isnt coded yet.'))

	# apply normalization to all matrices (all sets)
	for feature in input_features:
		for _set in sets_in_use:
			pt = X_stats[feature]['power_transformer']
			X_data_dict[(_set, feature)] = pt.transform(X_data_dict[(_set, feature)])

	return X_data_dict, scaler_settings
