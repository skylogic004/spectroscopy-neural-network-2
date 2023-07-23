import logging
from colorama import Fore, Back, Style
import colorama
import pickle
import pandas as pd
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
from os.path import join
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import linear_model, metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from spectra_ml.components.spectroscopy_NN import plotters
from spectra_ml.components.data_loader.data_structure import CurrentData

__author__ = "Matthew Dirks"

rmse = lambda arr1, arr2: np.sqrt(mean_squared_error(arr1, arr2))

# Setting CAP_NEGATIVE_PREDICTIONS to True assumes that negative predictions are NOT plausible in this domain!
# I used to do this later upon post-processing the results, but it simplifies the code to do it early
CAP_NEGATIVE_PREDICTIONS = True

def roundup(x):
	return int(math.ceil(x / 100.0)) * 100

def fitModel(Xtrain, Ytrain, model_name = 'OLS', model_params=None, normalize=False, fit_intercept=True, useStandardScaler=False):
	"""
	Args:
		normalize: "True" computes faster, but worse MAE when using all 1024 features. Same MAE when using <5 features. Note: StandardScaler also normalizes and is in use here.
		fit_intercept: Despite manually adding intercept to mldata, setting this to True still improves performance in some cases
	"""
	logger = logging.getLogger('spectra_ml')

	if (isinstance(Xtrain, pd.DataFrame)):
		Xtrain = Xtrain.values
	if (isinstance(Ytrain, pd.DataFrame)):
		Ytrain = Ytrain.values

	pipeline_steps = []

	# This is equivalent (I tested it) to my standardization (i.e. setting X_type=mean_std)
	# So I'll keep useStandardScaler=False always.
	if (useStandardScaler):
		logger.info('fitModel is using StandardScaler')
		prep_ss = StandardScaler()
		pipeline_steps.append(('ss', prep_ss))

	# get model instance
	if (model_name == 'OLS' or model_name == 'LEAST_SQUARES'):
		model = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
	elif (model_name.upper() == 'LASSO_ALPHA' or model_name.upper() == 'LASSO'):
		model = linear_model.Lasso(alpha=model_params['alpha'], max_iter=500000, fit_intercept=fit_intercept, normalize=normalize)
	elif (model_name == 'ridge'):
		model = linear_model.Ridge(max_iter=10000, fit_intercept=fit_intercept, normalize=normalize)
	elif (model_name == 'elasticnet'):
		model = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.8, max_iter=100000, fit_intercept=fit_intercept, normalize=normalize)
	elif (model_name == 'lassolars'):
		model = linear_model.LassoLars(alpha=0.1, fit_intercept=fit_intercept, normalize=normalize)
	elif (model_name == 'bayesianridge'):
		model = linear_model.BayesianRidge(tol=1e-5, fit_intercept=fit_intercept, normalize=normalize)
	elif (model_name == 'PLS'):
		assert not fit_intercept
		model = PLSRegression(n_components=model_params['n_components'], scale=False)
	elif (model_name == 'predict_the_average'):
		model = DummyRegressor(strategy='mean')
	elif (model_name == 'PCR'):
		# PCA followed by regression (via Least Squares)
		# e.g. https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py
		pca = PCA(n_components=model_params['n_components'], whiten=model_params['PCA_whiten'])
		pipeline_steps.append(('PCA', pca))

		model = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
	else:
		raise(ValueError('Unknown model name ({})'.format(model_name)))

	# Add model to the pipeline
	pipeline_steps.append(('model', model))

	# build the pipeline
	pipe = Pipeline(pipeline_steps)

	# train the pipeline
	pipe.fit(Xtrain, Ytrain)

	def printvals(arr):
		print('\t[0:5] = ', ','.join([str(x) for x in arr[0:5]]))
		print('\t[300:305] = ', ','.join([str(x) for x in arr[300:305]]))

	if (model_name != 'predict_the_average'):
		# print('>>>> coefs (internal). useStandardScaler = ' + str(useStandardScaler))
		# printvals(model.coef_)

		print('model.coef_ shape: ', model.coef_.shape)
		# ensure intercept and coefs have 2 dimensions always (to support multiple targets)
		# (if using 1 target, this will have an extra dimension of size 1)
		_coefs = model.coef_
		if (len(_coefs.shape) == 1):
			_coefs = _coefs.reshape(1, _coefs.shape[0])
			print('reshaped _coefs shape: ', _coefs.shape)

		if (fit_intercept):
			_intercepts = model.intercept_
			print('model.intercept_ shape: ', _intercepts.shape)
			assert len(_intercepts.shape) != 0
			if (len(_intercepts.shape) == 1):
				_intercepts = _intercepts.reshape(_intercepts.shape[0], 1)
				print('reshaped _intercepts shape: ', _intercepts.shape)
			assert _intercepts.shape[0] == _coefs.shape[0], 'shapes are {} and {} with num dimensions {} and {}'.format(_intercepts.shape, _coefs.shape, len(_intercepts.shape), len(_coefs.shape))

		transformed_coefs = None

		model_coefs = np.array(_coefs)
		if (model_name == 'PLS'):
			model_coefs = model_coefs.T

		# add intercept to list of coefs (because I consider there to be a column of 1s in the data)
		if (fit_intercept):
			model_coefs = np.concatenate([_intercepts, _coefs], axis=1)

		# sanity checks
		n_NaNs = np.isnan(model_coefs).sum()
		if (n_NaNs > 0):
			raise(Exception('model_coefs has {} NaNs'.format(n_NaNs)))

		if (transformed_coefs is not None):
			n_NaNs = np.isnan(transformed_coefs).sum()
			if (n_NaNs > 0):
				raise(Exception('transformed_coefs has {} NaNs'.format(n_NaNs)))
	else:
		transformed_coefs = model_coefs = None

	return model, transformed_coefs, model_coefs, pipe

def fit_multiblock_model(Xtrain_list, Ytrain, model_name, model_params):
	logger = logging.getLogger('spectra_ml')

	n_components = model_params['n_components']

	if (isinstance(Ytrain, pd.DataFrame)):
		Ytrain = Ytrain.values

	# pipeline_steps = []

	if (model_name == 'MB-PLS'):
		# https://mbpls.readthedocs.io/en/latest/
		from mbpls.mbpls import MBPLS
		model = MBPLS(n_components=n_components, standardize=False)
	else:
		raise(ValueError('Unknown model name ({})'.format(model_name)))

	# Add model to the pipeline
	# pipeline_steps.append(('model', model))

	# build the pipeline
	# pipe = Pipeline(pipeline_steps)
	pipe = model

	# train the pipeline
	model.fit(Xtrain_list, Ytrain)

	# model.plot(num_components=n_components) # generates multiple figures and displays. No save option. see https://github.com/DTUComputeStatisticsAndDataAnalysis/MBPLS/blob/master/mbpls/mbpls.py

	transformed_coefs = model_coefs = None # not using

	return model, transformed_coefs, model_coefs, pipe


def train_sklearn_model(
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
		model_name=None, # PLS or etc
		model_dump_fpath=None, # for loading a previously-saved model (which are saved below, with name like "model_dump_k#.pkl")
		model_params=None, # e.g. for PLS, {"n_components": 1234}
	):
	logger = logging.getLogger('spectra_ml')

	doing_multisensor = model_name in ['MB-PLS']
	if (doing_multisensor):
		logger.info(f'doing_multisensor because model is {model_name}')

	####################### CHECKING COMMAND ARGUMENTS #######################
	if (model_name == 'LASSO'):
		assert 'alpha' in model_params
	elif (model_name == 'PLS'):
		assert 'n_components' in model_params
	elif (model_name == 'MB-PLS'):
		assert 'n_components' in model_params
	elif (model_name == 'LEAST_SQUARES'):
		assert model_params is None

	##################### load pre-saved model, if any #################################
	if (model_dump_fpath is not None):
		logger.info(Fore.MAGENTA+'Loading model from disk ({}). NO TRAINING WILL BE PERFORMED.'.format(model_dump_fpath)+Fore.RESET)
		with open(model_dump_fpath, 'rb') as f:
			model_dump = pickle.load(f)
	else:
		model_dump = None

	##################### load data #################################
	assert 'use_dev' not in fold_spec or not fold_spec['use_dev'], 'These models can\'t use a dev set; set `use_dev` to False'
	cur_data = CurrentData(kth_fold, 
	                       dataset_dict,
	                       fold_spec,
	                       input_features,
	                       scaler_settings, 
	                       combine_train_dev=False)

	X_data_dict = cur_data.get_prepared_X()
	Xtrain_list = [X_data_dict[('train', feature)] for feature in input_features]
	Ytrain = cur_data.get_prepared_Y('train', target_columns_in_use)

	if (model_dump is not None): # will override the model, no training will be performed
		model = model_dump['model']
		transformed_coefs = model_dump['transformed_coefs']
		model_coefs = model_dump['model_coefs']
		pipe = model_dump['pipe']
	else:
		if (doing_multisensor):
			model, transformed_coefs, model_coefs, pipe = fit_multiblock_model(Xtrain_list, Ytrain, model_name, model_params)
		else:
			assert len(Xtrain_list)==1, 'MULTI SENSOR NOT SUPPORTED'
			Xtrain = Xtrain_list[0]
			fit_intercept = model_name != 'PLS'

			model, transformed_coefs, model_coefs, pipe = fitModel(
				Xtrain, 
				Ytrain, 
				model_name, 
				model_params, 
				fit_intercept=fit_intercept,
			)


	############################# POST-TRAINING ANALYSIS #####################################
	result_info = OrderedDict()

	# TODO: CODE DE-DUPLICATION. THIS CODE IS REPEATED (almost identical with some differences) TO: 
	#       dynamic_model.py, train_ROSA_model.py, train_sklearn_model.py
	# TODO: the df at the end I could have made all at once - without the concat - by making predictions on the whole thing
	df_per_set = []
	for which_set in cur_data.sets:
		# get ground truth
		groundtruth_df = cur_data.get(which_set)[target_columns_in_use].copy()
		assert groundtruth_df.index.name == 'sampleId'
		groundtruth_df['set'] = which_set

		# make predictions (these will be normalized, if normalization is in use)
		if (doing_multisensor):
			X = [X_data_dict[(which_set, feature)] for feature in input_features]
		else:
			X = X_data_dict[(which_set, input_features[0])]

		predictions_normalized = pipe.predict(X)

		# Convert numpy array into DataFrame.
		# Note: column names must match name used in `normalize_y_data.py` for unnormalization function to work
		predictions_normalized_df = pd.DataFrame(predictions_normalized, columns=target_columns_in_use)

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
		result_info[f'RMSE_{which_set}'] = score

		# rename columns from "Target" to "Target_pred" in the predictions DataFrame
		predictions_unnormalized_df.rename(columns={name:f'{name}_pred' for name in predictions_unnormalized_df.columns}, inplace=True)

		# copy the index from groundtruth_df
		predictions_unnormalized_df.index = groundtruth_df.index

		# merge predictions with groundtruth into one DataFrame
		# df_per_set.append(pd.merge(groundtruth_df, predictions_unnormalized_df, left_index=True, right_index=True))
		# save predictions and groundtruth
		df_per_set.append(groundtruth_df.join(predictions_unnormalized_df))

	# combine the dataframes from each set
	df = pd.concat(df_per_set)

	# save predictions to pkl
	logger.info('Saving best_model_predictions.pkl...')
	predictions_dict = {
		'df': df,
		'target_columns': target_columns_in_use,
		'prediction_columns': [f'{target}_pred' for target in target_columns_in_use],
		'kth_fold': kth_fold,
	}
	with open(join(out_dir, 'best_model_predictions.pkl'), 'wb') as f:
		pickle.dump(predictions_dict, f)

	# plot predictions
	if (configData['plot']['target_vs_predictions']):
		logger.info('Plotting final version of targets vs predictions (individual plots)')

		plotters.plot_target_vs_predictions_individual(
			predictions_dict['df'],
			target_columns_in_use,
			join(out_dir, 'prediction_plots'), 
		)

	# save human-readable result info
	with open(join(out_dir, 'result_info.json'), 'w') as text_file:
		text_file.write(json.dumps(result_info, indent=2))


	# plot coefficients
	if (model_coefs is not None and configData['sklearn_models']['plot_coefs']):
		if (len(input_features) == 1):
			input_columns = cur_data.feature_columns[input_features[0]]
		else:
			raise(ValueError('This code is only for single-sensor models'))

		try:
			logger.info('Plotting coefficients... (model_coefs.shape = {})'.format(model_coefs.shape))

			if (fit_intercept):
				# tmp_columns = ['bias']+features_flattened
				tmp_columns = ['bias']+input_columns
			else:
				# tmp_columns = features_flattened
				tmp_columns = input_columns

			model_coefs_df = pd.DataFrame(model_coefs, index=target_columns_in_use, columns=tmp_columns)
			model_coefs_df.to_csv(join(out_dir, 'model_coefs_df.csv'))

			for assay_col, row in model_coefs_df.iterrows():
				n = len(input_columns)
				x = range(n)
				n_ticks = 30
				stepsize = roundup(n / n_ticks)

				fig, ax = plt.subplots(1, 1, figsize=(15,10))
				ax.plot(x, row[input_columns])
				ax.set_xticks(x[::stepsize])
				ax.set_xticklabels(input_columns[::stepsize], rotation=20)

				if ('bias' in row):
					ax.set_title(f'bias={row["bias"]}')

				fig.tight_layout()
				fig.savefig(join(out_dir, f'model_coefs_{assay_col}.png'))


		except Exception as e:
			logger.info('Failed to plot coefficients due to exception. Exception: {}'.format(e))
			# traceback.print_tb(e.__traceback__)

	# plot loadings
	if (configData['sklearn_models']['plot_loadings']):
		if (len(input_features) == 1):
			input_columns = cur_data.feature_columns[input_features[0]]
		else:
			raise(ValueError('This code is only for single-sensor models'))

		if (hasattr(model, 'x_loadings_')):
			loadings = model.x_loadings_.T
			n = len(input_columns)
			x = range(n)
			n_ticks = 30
			stepsize = roundup(n / n_ticks)

			fig, axs = plt.subplots(loadings.shape[0], 1, figsize=(15,10*loadings.shape[0]))
			for idx, (ax, loading) in enumerate(zip(axs, loadings)):
				ax.plot(x, loading, c='k')
				ax.set_xticks(x[::stepsize])
				ax.set_xticklabels(input_columns[::stepsize], rotation=20)
				ax.set_title(f'Component {idx}')

			fig.tight_layout()
			fig.savefig(join(out_dir, 'loadings.png'))

	return predictions_dict