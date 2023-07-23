import logging
import pandas as pd
import numpy as np
from collections import OrderedDict

from spectra_ml.components.data_loader.normalize_y_data import scale_via_min_max, scale_via_mean_std, target_min_max_unnormalize, target_mean_std_unnormalize
from spectra_ml.components.data_loader.normalize_x_data import scale_x_via_max, scale_x_via_min_max, scale_x_via_mean_std, scale_x_via_PowerTransformer

__author__ = "Matthew Dirks"

class CurrentData():
	""" Holds data for the current fold or bootstrap (train, dev, and test sets for current training run) """

	def __init__(self, kth_fold, dataset_dict, fold_spec, input_features, scaler_settings={'X_type': 'none', 'Y_type': 'none'}, combine_train_dev=False, X_dtype=np.float32, log_fn=None):
		self.log_fn = log_fn if log_fn is not None else logging.getLogger('spectra_ml').info
		self.partitions = self.get_kth_fold(kth_fold, dataset_dict['shuffled_df'], fold_spec)

		# self.sample_primary_key = 40000 # sample IDs will start from this number
		self.combine_train_dev = combine_train_dev

		self.sets = list(self.partitions.keys())
		if (combine_train_dev and 'dev' in self.sets): 
			# remove dev set, because it will be combined with train
			self.sets.remove('dev')

		self.input_features = input_features # array of strings, like ['XRF', 'HYPER']. These are the names of the "input features" or "blocks" (as they're known in Chemometrics). In my case, these are the names of different spectra types.
		self.feature_columns = dataset_dict['feature_columns'] # OrderedDict, where keys are input features and values are lists of column names
		self.target_columns = dataset_dict['target_columns_dict'].values()

		# update index of all DataFrames with unique IDs
		for _set, df in self.partitions.items():
			self.update_index_with_unique_IDs(df)

		# normalize the data (if requested)
		self.scaler_settings = scaler_settings
		self.prepare_X_data_dict(X_dtype)
		self.prepare_Y_data_dict(self.target_columns)

	def update_index_with_unique_IDs(self, df):
		for col_name in ['sampleId', 'index']:
			if (col_name in df):
				# use this for sample ID index
				df.set_index(col_name, inplace=True)
				df.index.name = 'sampleId'
				return
		raise Exception('df needs to have a unique sample ID for the index')


	def keys(self, include_unsuper=True):
		keys = list(self.partitions.keys())

		has_unsuper = 'unsuper' in keys

		if (has_unsuper):
			keys.remove('unsuper')

		if (has_unsuper and include_unsuper):
			# this appends it to the end when included; always at the end
			keys.append('unsuper')

		return keys

	def set_unsuper(self, df):
		self.update_index_with_unique_IDs(df)
		self.partitions['unsuper'] = df

	def get(self, _set):
		""" Get data from partitions and if combine_train_dev requested, then combine data from train set and dev set """
		# get all sets (concat them)
		# if (_set == 'ALL'):
			# dfs = [self.get(which_set) for which_set in 

		if (not self.combine_train_dev):
			# this is the typical usage (combine_train_dev=False)
			return_df = self.partitions[_set]
		else:
			if (_set == 'dev'):
				raise(ValueError('dev set not available. Since combine_train_dev requested, dev set will be moved into train set.'))

			elif (_set == 'train'):
				# combine train and dev sets
				train_df = self.partitions['train']
				dev_df = self.partitions['dev']

				return_df = pd.concat([train_df, dev_df], axis=0)

				# sanity check
				assert return_df.shape[1] == train_df.shape[1] == dev_df.shape[1]
				assert return_df.shape[0] == train_df.shape[0] + dev_df.shape[0]

			elif (_set in ['test', 'test_CV']):
				# combine_train_dev doesn't affect test set; return it as-is
				return_df = self.partitions['test']

		if (return_df is not None and return_df.shape[0] > 0):
			return return_df
		else:
			return None

	def prepare_Y_data_dict(self, target_columns):
		""" Get the "Y" data and prepare it for use by TF (convert to numpy array, float32, and normalize if requested) """

		Y_type = self.scaler_settings['Y_type']

		if (Y_type == 'none'):
			Y_data_dict = OrderedDict()
			for _set in self.sets:
				Y_data_dict[_set] = self.get(_set)[target_columns]
			Y_normalization_stats = None
		elif (Y_type == 'min_max'):
			Y_data_dict, Y_normalization_stats = scale_via_min_max(self.scaler_settings, self, target_columns, self.sets)
		elif (Y_type == 'mean_std'):
			Y_data_dict, Y_normalization_stats = scale_via_mean_std(self.scaler_settings, self, target_columns, self.sets)
		else:
			raise(ValueError('Must provide a valid type of normalization to use on Y data via `scaler_settings["Y_type"]`'))

		# Also convert to float32 (for TF mostly)
		for key, df in Y_data_dict.items():
			Y_data_dict[key] = df.astype(np.float32)

		self.Y_data_dict = Y_data_dict
		self.Y_normalization_stats = Y_normalization_stats

	def get_prepared_Y(self, which_set, target_columns_in_use):
		""" Returns data that is converted to float32 and possibly normalized, if requested """
		assert not self.combine_train_dev, 'This feature isn\'t coded yet'
		return self.Y_data_dict[which_set][target_columns_in_use]

	def unnormalize_Y_df(self, df):
		Y_type = self.scaler_settings['Y_type']
		if (Y_type == 'none'):
			return df # do nothing
		elif (Y_type == 'min_max'):
			return df.apply(target_min_max_unnormalize, axis=0, args=(self.Y_normalization_stats,))
		elif (Y_type == 'mean_std'):
			return df.apply(target_mean_std_unnormalize, axis=0, args=(self.Y_normalization_stats,))

	def prepare_X_data_dict(self, X_dtype):
		""" Get the "X" data and prepare it for use by TF (or other models).
		Namely: 
		- convert to numpy array
		- TF likes float32, ROSA likes float64
		- normalize if requested
		"""
		X_type = self.scaler_settings['X_type']

		# importance scales (scales data from one sensor or "feature" more/less than the others)
		importance_scales = None # NOT IN USE CURRENTLY
		if (importance_scales is None):
			importance_scales = [1] * len(self.input_features)
		assert len(importance_scales)==len(self.input_features)

		# check for valid arguments
		if (X_type in ['min_max', 'mean_std', 'none', 'power_transformer']):
			assert all([val==1 for val in importance_scales]), 'importance_scales not implemented for this scaler type'

		# Grab the X data, grouped by set and input_feature, then convert to numpy array, cast as float32, and convert to 2D matrix (not vector) if needed. 
		# e.g. X_data_dict will have keys like one of these: ('train', 'XRF'), ('dev', 'XRF'), ('test', 'XRF')
		X_data_dict = OrderedDict()
		for _set in self.sets:
			# get data for kth fold, train/test/dev set
			_df = self.get(_set)

			# setup X data (with all input_features)
			for feature in self.input_features:
				# from _df, select input features which belong to sensor type(s) like XRF or HYPER
				# and convert to numpy matrix of floats, for tensorflow
				if (_df is not None):
					X_data_dict[(_set, feature)] = _df[self.feature_columns[feature]].values.astype(X_dtype)
					self.log_fn('Number of samples in ({}, {}) set: {}'.format(_set, feature, _df.shape[0]))
				else:
					self.log_fn('Number of samples in ({}, {}) set: None (disabled)'.format(_set, feature))

				# ensure X is a matrix not a vector
				if (len(X_data_dict[(_set, feature)].shape) == 1):
					X_data_dict[(_set, feature)] = X_data_dict[(_set, feature)].reshape(-1, 1)

		# Do requested normalization
		if (X_type == 'max'):
			X_data_dict, self.scaler_settings = scale_x_via_max(self.scaler_settings, X_data_dict, self.sets, self.input_features, importance_scales)
		elif (X_type == 'min_max'):
			X_data_dict, self.scaler_settings = scale_x_via_min_max(self.scaler_settings, X_data_dict, self.sets, self.input_features)
		elif (X_type == 'mean_std'):
			X_data_dict, self.scaler_settings = scale_x_via_mean_std(self.scaler_settings, X_data_dict, self.sets, self.input_features)
		elif (X_type == 'power_transformer'):
			X_data_dict, self.scaler_settings = scale_x_via_PowerTransformer(self.scaler_settings, X_data_dict, self.sets, self.input_features)
		elif (X_type == 'none'):
			self.log_fn('X is not normalized.')
		else:
			raise(ValueError('Must provide a valid type of normalization to use on X data via `scaler_settings["X_type"]`'))

		self.X_data_dict = X_data_dict

	def get_prepared_X(self):
		""" Returns data that is converted to float32 (or whatever X_dtype is) and possibly normalized, if requested """
		return self.X_data_dict




	def get_kth_fold(self, kth_fold, df, fold_spec):
		"""
		Args:
			fold_spec:
				n_super: number of samples to use for supervised learning (will be further split into train, dev, test sets)
				n_unsuper: number of samples to use for unsupervised learning
				n_dev: number of samples to use for dev set; will be selected from the training set which is from the supervised set of data
				random_state: for df.sample
				r: number of CV repeats
				k: number of CV folds
		"""
		assert all([key not in fold_spec for key in ['how_to_use_unassigned', 'n_unsuper']]), 'This feature not supported anymore'

		# prepare data to actually use for this particular instance of kth_fold
		# it should have these keys: train, dev, test
		partitions = OrderedDict()

		if (fold_spec['type'] == '70/10/20_split'):
			# check that settings are valid
			assert all([key in fold_spec for key in ['type']])
			assert kth_fold == 0

			# Pandas Series naming which set each sample is in (e.g. "train", "test", etc)
			assignments = df[fold_spec['type']] 

			# use the data splits specified in column
			partitions['train'] = df[assignments=='train']
			partitions['dev'] = df[assignments=='dev']
			partitions['test'] = df[assignments=='test']

		elif (fold_spec['type'] in ['DUPLEX_split', '10fold_and_test_split']):
			# check that settings are valid
			assert all([key in fold_spec for key in ['type', 'use_dev']])
			assert kth_fold >= 0 and kth_fold <= 9

			# Pandas Series naming which set each sample is in (e.g. "train") or which fold ("partition_0" etc)
			assignments = df[fold_spec['type']] 
			
			# Test set is labelled 'test':
			partitions['test'] = df[assignments=='test']

			# Held-out set for the current fold (iteration) of CV:
			partition_name = f'partition_{kth_fold}'
			partitions['test_CV'] = df[assignments==partition_name]

			# get remaining samples (NOT in test and NOT in test_CV)
			remainder_df = df[~assignments.isin([partition_name, 'test'])]
			if (fold_spec['use_dev']):
				# Dev set (here, it's only used for early-stopping by neural networks) will be a *random* selection EACH TIME the model is trained
				dev_seed = fold_spec.get('seed', None)
				dev_df = remainder_df.sample(n=14, replace=False, random_state=dev_seed)
				partitions['dev'] = dev_df

				self.log_fn(f'dev set randomly sampled (seed: {dev_seed})')
				# sampleIds are {dev_df["sampleId"].tolist()}

				# everything else will be for training
				train_df = remainder_df[~remainder_df.index.isin(dev_df.index)]
				partitions['train'] = train_df

				# assert len(set(train_df.index).union(set(dev_df.index)).union(set(partitions['test'].index)).union(set(partitions['test_CV'].index))) == 177
			else:
				partitions['train'] = remainder_df

		else:
			raise(ValueError('fold_spec type supported.'))

		return partitions
