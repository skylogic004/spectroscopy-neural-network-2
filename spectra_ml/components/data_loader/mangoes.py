import logging
import pandas as pd
from sklearn.utils import shuffle
from colorama import Fore, Back, Style
from collections import OrderedDict
from scipy.signal import savgol_filter
import numpy as np
import datetime
import random
import math
from sklearn.model_selection import LeaveOneOut, LeavePOut, ShuffleSplit, KFold, RepeatedKFold

CSV_FPATH = "data/NAnderson2020MendeleyMangoNIRData.csv"
MAT_FPATH = "data/mango_dm_full_outlier_removed2.mat"
METADATA_FPATH = "data/metadata.pkl"
RANDOM_STATE = 5000

def load_Anderson_data():
	logger = logging.getLogger('spectra_ml')
	logger.info(Fore.CYAN + 'Dataset: Mango' + Fore.RESET)

	# load each sensor
	df, feature_columns, targets_to_use = load_Anderson_data_helper()

	target_columns_dict = {x:x for x in targets_to_use}

	# random shuffle
	shuffled_df = shuffle(df, random_state=RANDOM_STATE).copy()
	shuffled_df.reset_index(drop=False, inplace=True) # let index be row count from 0 onward (in the new shuffled state)

	# not in use
	shuffled_df['ignore'] = False


	return {
		'shuffled_df': shuffled_df, 
		'target_columns_dict': target_columns_dict, 
		'feature_columns': feature_columns,
		'extra_columns': ['Date'],
	}

def load_Anderson_data_helper():
	logger = logging.getLogger('spectra_ml')

	# spectra
	df = pd.read_csv(CSV_FPATH)
	nir_columns = df.columns[list(df.columns).index('285'):]
	assert len(nir_columns) == 306

	# assays
	targets_to_use = ['DM']

	# original dataset split into sets -- rename the values
	# mapper = {'Cal': 'calibrate', 'Tuning': 'tuning', 'Val Ext': 'test'}
	# df['rand_split'] = df['Set'].apply(mapper.get)

	# create new validation split(s)
	# df['datetime'] = pd.to_datetime(df['Date'])
	# season_to_set = {1:'calibrate', 2:'calibrate', 3:'validate', 4:'test'}
	# df.loc[:, 'D2017_split'] = df['Season'].apply(season_to_set.get)

	# use the "Val Ext" set from the original dataset as the test set,
	# then make a new 10-fold cross-validation partitioning on the remaining data
	split_name = '10fold_and_test_split'
	df[split_name] = 'train'
	df.loc[df['Set']=='Val Ext', split_name] = 'test'

	# with the remaining samples (y)...
	y = df[df[split_name]!='test'].index

	# do 10-fold CV on y
	kf = KFold(n_splits=10)
	for k, (cv_train, cv_test) in enumerate(kf.split(y)):
		df.loc[y[cv_test], split_name] = f'partition_{k}'

	logger.info(f'Datset partitioned into sets; number of samples in each is:\n{df[split_name].value_counts()}')


	# Paper says they use 103 features but there are 306 bins in the spectra. They said they use 684 to 990 nm, so lets truncate to match
	wavelengths = [int(x) for x in nir_columns]
	wavelengths_truncated = [x for x in wavelengths if (x >= 684 and x <= 990)]
	nir_columns_truncated = [str(x) for x in wavelengths_truncated]

	# Meta-data: what are the column names
	feature_columns = OrderedDict()
	feature_columns['NIR'] = nir_columns
	feature_columns['NIR_truncated'] = nir_columns_truncated

	# Pre-processing
	df, feature_columns = do_preprocessing(feature_columns, nir_columns_truncated, df)

	return df, feature_columns, targets_to_use

def snv(spectrum):
	"""
	See https://towardsdatascience.com/scatter-correction-and-outlier-detection-in-nir-spectroscopy-7ec924af668,

	:snv: A correction technique which is done on each
	individual spectrum, a reference spectrum is not
	required
	
	return:
		Scatter corrected spectra
	"""
	return (spectrum - np.mean(spectrum)) / np.std(spectrum)

def do_preprocessing(feature_columns, nir_columns_truncated, df):
	# SNV
	snv_df = df[nir_columns_truncated].apply(snv, axis=1).rename(columns={col:f'SNV_{col}' for col in nir_columns_truncated})
	feature_columns['SNV'] = snv_df.columns.tolist()

	# 1st DERIVATIVE
	window_length = 13
	polyorder = 2

	deriv = 1
	columns = [f'SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SG{deriv}'] = columns

	sg_1stderiv_df = pd.DataFrame(df[nir_columns_truncated].apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)

	# 2nd DERIVATIVE
	deriv = 2
	columns = [f'SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SG{deriv}'] = columns

	sg_2ndderiv_df = pd.DataFrame(df[nir_columns_truncated].apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)


	# 1st DERIVATIVE on SNV
	deriv = 1
	columns = [f'SNV_SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SNV_SG{deriv}'] = columns

	snv_sg_1stderiv_df = pd.DataFrame(snv_df.apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)

	# 2nd DERIVATIVE on SNV
	deriv = 2
	columns = [f'SNV_SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SNV_SG{deriv}'] = columns

	snv_sg_2ndderiv_df = pd.DataFrame(snv_df.apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)

	# combine dfs
	combined_df = pd.concat([df, snv_df, sg_1stderiv_df, sg_2ndderiv_df, snv_sg_1stderiv_df, snv_sg_2ndderiv_df], axis=1)
	assert df.shape[0] == combined_df.shape[0] == snv_df.shape[0] == snv_sg_2ndderiv_df.shape[0]

	# check sizes
	total = 0
	for key, value in feature_columns.items():
		print(key, len(value))
		total += len(value)
	print('Total number of features: ', total)

	return combined_df, feature_columns
