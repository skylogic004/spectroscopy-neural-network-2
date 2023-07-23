import logging
import pandas as pd
import numpy as np
from collections import OrderedDict

def target_min_max_normalize(column_series, Y_normalization_stats):
	_min = Y_normalization_stats.loc[column_series.name, 'min']
	_max = Y_normalization_stats.loc[column_series.name, 'max']
	if (_max == _min): # range is 0, so just divide by max instead to make all values 1 (avoids divide-by-0 error)
		if (_max == 0):
			div = 1 # do nothing if ALL values are 0
		else:
			div = _max
	else:
		div = _max - _min
	return (column_series - _min) / div

def target_min_max_unnormalize(column_series, Y_normalization_stats):
	_min = Y_normalization_stats.loc[column_series.name, 'min']
	_max = Y_normalization_stats.loc[column_series.name, 'max']
	if (_max == _min): # avoid divide-by-0 error
		div = _max
	else:
		div = _max - _min
	return column_series * div + _min

def target_mean_std_normalize(column_series, Y_normalization_stats):
	_mean = Y_normalization_stats.loc[column_series.name, 'mean']
	_std = Y_normalization_stats.loc[column_series.name, 'std']
	return (column_series - _mean) / _std

def target_mean_std_unnormalize(column_series, Y_normalization_stats):
	_mean = Y_normalization_stats.loc[column_series.name, 'mean']
	_std = Y_normalization_stats.loc[column_series.name, 'std']
	return column_series * _std + _mean

'''
def setup_y_data_WITHOUT_NORMALIZING(cur_data,
									 target_columns,
									 sets_in_use):
	# these structures will hold data in the numpy array dtype
	Y_data_dict = OrderedDict()
	for _set in sets_in_use:
		# get data for kth fold, train/test/dev set
		_df = cur_data.get(_set)

		# setup Y data
		if (_set != 'unsuper'):
			# Y_data_dict[_set] = normalize_Y_df(_df[target_columns]).values
			Y_data_dict[_set] = _df[target_columns]

		if (_set == 'train'):
			# sanity check
			for col in target_columns:
				assert _df[col].isnull().sum() == 0, 'Expected no nulls, but found number of nulls >= 0 in column {}'.format(col)

	return Y_data_dict, target_nop_unnormalize
'''

def scale_via_min_max(scaler_settings,
					  cur_data, 
					  target_columns, 
					  sets_in_use,
					  ):
	""" Prepare data (normalize, convert to float32) for use by tensorflow or any ML model training.
	This is considered to be "low level" compared to the CurrentData structure.
	"""
	logger = logging.getLogger('spectra_ml')

	# normalize assay (scale to between 0 and 1), because input data range is very small for some columns
	# First, get training data (we normalize to training data, NOT to test set)
	train_assay_df = cur_data.get('train').loc[:, target_columns]

	# calc stats
	Y_normalization_stats = pd.DataFrame({'min': train_assay_df.min(), 'max': train_assay_df.max()})
	logger.info('Target data min-max normalized on training set using these min & max values: \n{}'.format(Y_normalization_stats.to_string()))

	# prepare normalization function to apply to all target columns
	normalize_Y_df = lambda df: df.apply(target_min_max_normalize, axis=0, args=(Y_normalization_stats,))
	assert normalize_Y_df(train_assay_df).isnull().sum().sum() == 0, 'Expected no nulls, but found number of nulls >= 0 in normalized data.\nNOTE: Y_normalization_stats={}'.format(Y_normalization_stats)

	# these structures will hold *normalized* data
	Y_data_dict = OrderedDict()
	for _set in sets_in_use:
		_df = cur_data.get(_set)

		# setup Y data
		Y_data_dict[_set] = normalize_Y_df(_df[target_columns])

	# sanity-check: make sure normalizing did in fact set range to [0, 1] (on training data only)
	assert (Y_data_dict['train'].min()==0).all(), 'Normalization sanity-check failed'
	assert (Y_data_dict['train'].max()==1).all(), 'Normalization sanity-check failed'

	return Y_data_dict, Y_normalization_stats

def scale_via_mean_std(scaler_settings,
					   cur_data, 
					   target_columns, 
					   sets_in_use,
					   ):
	""" Prepare data (normalize, convert to float32) for use by tensorflow or any ML model training.
	This is considered to be "low level" compared to the CurrentData structure.
	"""
	# logger = logging.getLogger('spectra_ml')

	# First, get training data (we normalize to training data, NOT to test set)
	train_assay_df = cur_data.get('train').loc[:, target_columns]

	# calc stats
	Y_normalization_stats = pd.DataFrame({'mean': train_assay_df.mean(axis=0), 'std': train_assay_df.std(axis=0)})
	# logger.info(f'Target data normalized via mean_std on training set: \n{Y_normalization_stats}')

	# prepare normalization function to apply to all target columns
	normalize_Y_df = lambda df: df.apply(target_mean_std_normalize, axis=0, args=(Y_normalization_stats,))
	assert normalize_Y_df(train_assay_df).isnull().sum().sum() == 0, f'Expected no nulls, but found number of nulls >= 0 in normalized data.\nNOTE: Y_normalization_stats={Y_normalization_stats}'

	# these structures will hold *normalized* data
	Y_data_dict = OrderedDict()
	for _set in sets_in_use:
		_df = cur_data.get(_set)

		# setup Y data
		Y_data_dict[_set] = normalize_Y_df(_df[target_columns])

	return Y_data_dict, Y_normalization_stats
