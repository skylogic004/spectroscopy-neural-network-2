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

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri, Formula
from rpy2.robjects.packages import importr

from spectra_ml.components.spectroscopy_NN import plotters
from spectra_ml.components.data_loader.data_structure import CurrentData

__author__ = "Matthew Dirks"

rmse = lambda arr1, arr2: np.sqrt(mean_squared_error(arr1, arr2))

# Setting CAP_NEGATIVE_PREDICTIONS to True assumes that negative predictions are NOT plausible in this domain!
# I used to do this later upon post-processing the results, but it simplifies the code to do it early
CAP_NEGATIVE_PREDICTIONS = True

def roundup(x):
	return int(math.ceil(x / 100.0)) * 100

from contextlib import contextmanager
from rpy2.robjects.lib import grdevices

@contextmanager
def r_plot(width=600, height=600, dpi=100):
	""" 
	Usage:
	  with r_plot():
		ro.r('some_plotting_function_in_R')(args)
	"""

	with grdevices.render_to_bytesio(grdevices.png, 
									 width=width,
									 height=height, 
									 res=dpi) as b:

		yield

	data = b.getvalue()
	# display(Image(data=data, format='png', embed=True))
	return data

def get_predictions(rosa_result, data_R):
	"""
	Args:
		rosa_results: The result object from fitting a ROSA model
		data_R: a rpy2.robjects.vectors.ListVector
	"""

	R_code = """
	function(object, newdata, ncomp = 1:object$ncomp) {
		newX1 <- model.frame(formula(object), data = newdata)
		newX2 <- newX1[-1] # removes the first column which was the TARGET column
		newX <- do.call(cbind,newX2)

		# number of observations:
		nobs <- dim(newX)[1]

		# Get the coefs for each component
		# (original code has intercept=TRUE but that gives me an error)
		# the shape of B_orig is (num_features, 1, num_components)
		B_orig <- coef(object, ncomp = ncomp, intercept = FALSE)

		# Get the average coef value
		# This is equivalent to calculating the prediction per component and averaging the results:
		# The shape of B is (num_features, 1)
		B <- rowMeans(B_orig, dims=2)

		# This next line is based on https://github.com/khliland/multiblock/blob/87e18bcf7d38e5b840830f63fbd51e2606e83770/R/rosa_results.R#L52
		# This calculates the amount to shift the output value by as the mean of Y minus the prediction given the mean of X
		B0 <- object$Ymeans - object$Xmeans %*% B

		# Calculate the prediction.
		# This is `newX.dot(B) + B0` in Python
		pred <- newX %*% B + rep(c(B0), each = nobs)

		return(pred)
	}
	"""
	get_predictions_R = ro.r(R_code)
	
	return np.array(get_predictions_R(rosa_result, data_R)).squeeze()

def train_ROSA_model(
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
		n_components=1,
		internal_validation=ro.r('NULL'),
	):
	logger = logging.getLogger('spectra_ml')

	##################### load data #################################
	assert 'use_dev' not in fold_spec or not fold_spec['use_dev'], 'These models can\'t use a dev set; set `use_dev` to False'
	cur_data = CurrentData(kth_fold, 
						   dataset_dict,
						   fold_spec,
						   input_features,
						   scaler_settings, 
						   combine_train_dev=False,
						   X_dtype=np.float64)

	X_data_dict = cur_data.get_prepared_X()

	##################### prepare data for R #################################
	data_R_dict = {}
	with ro.default_converter + numpy2ri.converter:
		py2rpy = ro.conversion.get_conversion().py2rpy
		
		for which_set in cur_data.sets:
			dict_for_r = {feature: py2rpy(X_data_dict[(which_set, feature)]) for feature in input_features}
			dict_for_r['TARGET'] = cur_data.get_prepared_Y(which_set, target_columns_in_use).values
			data_R_dict[which_set] = ro.ListVector(dict_for_r)

	##################### train the model #################################

	# prepare "Formula" for R
	block_names_formula = ' + '.join(input_features)
	formula = Formula(f'TARGET ~ {block_names_formula}')

	# train the model
	multiblock = importr('multiblock')
	# rosa_result = multiblock.rosa(formula, data=data_R_dict['train'], ncomp=n_components, internal_validation="CV10")
	rosa_result = multiblock.rosa(formula, data=data_R_dict['train'], ncomp=n_components, internal_validation=internal_validation)

	plot_loadings_R = ro.r("""
		library(grDevices)
		library(ggplot2)
		function(nc, result, fpath){
		  png(fpath, width=1000, height=500, units="px", res=120)
		  p <- loadingplot(result, scatter=FALSE, comps=1:nc)
		  dev.off()
		}
	""")

	# save = importr('save')
	# save.ggsave(filename=join(out_dir, "x.pdf"), plot=fig, width=200, height=120, unit='mm')
	plot_loadings_R(n_components, rosa_result, join(out_dir,'loadingplot.png'))

	# to help with getting results out, map the names of the properties
	name_to_idx = dict([(name, idx) for idx, name in enumerate(list(rosa_result.names))])

	# the winning blocks (and switch from 1-indexed to 0-indexed)
	idx_order = (np.array(rosa_result[name_to_idx['order']]).flatten()-1).tolist()

	logger.info(f'ROSA picked these blocks for the {n_components} components: {idx_order}')
	logger.info(f'which corresponds to these input_features: {np.array(input_features)[idx_order].tolist()}')

	# there's a "weight" for each of the features, times 3 (one for each component)
	# There's many zeroes here - good. Does this line up with 'order'?
	loadings = np.array(rosa_result[name_to_idx['loading.weights']]).reshape((-1, n_components)).T
	
	# plot matrix of loadings as an image
	fig, ax = plt.subplots(figsize=(10, n_components/3))
	im = ax.imshow(loadings, aspect='auto', interpolation='none', cmap='bone')
	ax.set_yticks(np.arange(loadings.shape[0]));
	ax.set_ylabel('Number of components')
	ax.set_xlabel(f'idx ({input_features})');
	fig.tight_layout()
	fig.savefig(join(out_dir, 'loadings_matrix.png'))

	# plot "Candidate Score Correlations" plot using the R package
	# "makes an image plot of each candidate score's correlation to the winner or the block-wise response residual."
	plot_candidate_score_cor_R = ro.r("""
		function(result, fpath) {
		  png(fpath, width=1000, height=400, units="px", res=120)
		  p <- image(result)
		  dev.off()
		}
	""")
	plot_candidate_score_cor_R(rosa_result, join(out_dir,'candidates.png'))

	# plot bar chart
	# "barplot.rosa makes barplot of block and component explained variances."
	plot_bar_R = ro.r("""
		function(result, fpath) {
		  png(fpath, width=500, height=300, units="px", res=120)
		  p <- barplot(result)
		  dev.off()
		}
	""")
	plot_bar_R(rosa_result, join(out_dir,'bar.png'))


	############################# POST-TRAINING ANALYSIS #####################################
	result_info = OrderedDict()

	# TODO: CODE DE-DUPLICATION. THIS CODE IS REPEATED (almost identical with some differences) TO: 
	#       dynamic_model.py, train_ROSA_model.py, train_sklearn_model.py
	df_per_set = []
	for which_set in cur_data.sets:
		# get ground truth
		groundtruth_df = cur_data.get(which_set)[target_columns_in_use].copy()
		assert groundtruth_df.index.name == 'sampleId'
		groundtruth_df['set'] = which_set

		# make predictions (these will be normalized, if normalization is in use)
		predictions_normalized = get_predictions(rosa_result, data_R_dict[which_set])

		# Convert numpy array into DataFrame.
		# Note: column names must match name used in `data_structure.py` for unnormalization function to work
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
	'''
	if (model_coefs is not None and configData['sklearn_models']['plot_coefs']):
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
	'''
	# plot loadings
	if (configData['sklearn_models']['plot_loadings']):
		loadings = np.array(rosa_result[name_to_idx['loading.weights']]).reshape((-1, n_components)).T

		input_feature_sizes = [X_data_dict[('train', feature)].shape[1] for feature in input_features]
		total_feature_count = sum(input_feature_sizes)
		feature_positions = np.cumsum([0]+input_feature_sizes)
		feature_positions

		# get order of winning blocks and switch from 1-indexed to 0-indexed:
		idx_order = (np.array(rosa_result[name_to_idx['order']])-1).flatten().tolist()

		# make plot:
		fig, axs = plt.subplots(n_components, 1, figsize=(15,10*n_components))
		for comp_idx, (ax, winning_idx, loading_for_all_blocks) in enumerate(zip(axs, idx_order, loadings)):
			i0 = feature_positions[winning_idx]
			i1 = feature_positions[winning_idx+1]
			loading = loading_for_all_blocks[i0:i1]
			ax.plot(loading)
			ax.set_title(f'Component {comp_idx}: {input_features[winning_idx]}')

		fig.tight_layout()
		fig.savefig(join(out_dir, 'loadings.png'))


	return predictions_dict