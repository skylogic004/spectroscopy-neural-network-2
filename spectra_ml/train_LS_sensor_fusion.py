from os.path import exists, join, isdir
import pandas as pd
from collections import OrderedDict
from sklearn import linear_model, metrics
import os
import logging
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime as dt

from spectra_ml.components.spectroscopy_NN import plotters
from spectra_ml.components.models.train_sklearn_model import rmse
from spectra_ml.components.prepare_out_dir_and_logging import prepare_out_dir_and_logging
import spectra_ml.components.colorama_helpers as _c
from spectra_ml.components.config_io.config_toml import getConfig, getConfigOrCopyDefaultFile
from spectra_ml.components.get_results_dir import get_results_dir

BEST_MODEL_PRED_FNAME = 'best_model_predictions.pkl'
CAP_NEGATIVE_PREDICTIONS = True
CONFIG_FPATH = "config.toml"
CONFIG_DEFAULT_FPATH = "config_DEFAULT.toml"

# Which set(s)'s predictions to use as "training" in the high-level fusion model:
# (other good options are ['test_CV'] or ['train', 'test_CV'])
# ('train' is for the sklearn models; 'train_or_dev' is for the NN models)
SETS_FOR_TRAINING = ['train', 'train_or_dev']

WHICH_MODEL = 'NNLS'
# WHICH_MODEL = 'OLS'

def process_each_fold(out_dir, kth_fold, input_dirs, configData):
	logger = logging.getLogger('spectra_ml')

	#### load all the pkls from each sensor (for the current fold)
	for ob in input_dirs:
		# predictions file will be in one of two places, depending on whether these model results 
		# were run locally or run on the compute cluster. 
		# So search both locations:
		fp_options = [
			join(ob['dir_path'], f'k={kth_fold}', BEST_MODEL_PRED_FNAME),
			join(ob['dir_path'], f'k={kth_fold}--{BEST_MODEL_PRED_FNAME}'),
		]
		for fp in fp_options:
			if (exists(fp)):
				ob['pkl_data'] = pd.read_pickle(fp)
				break
		if ('pkl_data' not in ob):
			raise(ValueError(f'Couldn\'t find predictions file for k={kth_fold} in {ob["dir_path"]}'))

	# Read in the column names of input (groundtruth) and output (model predictions)
	first_pkl = input_dirs[0]['pkl_data']
	try:
		target_column = first_pkl['assay_columns'][0]
	except:
		target_column = first_pkl['target_columns'][0]
	target_columns_in_use = [target_column]

	pred_col = first_pkl['prediction_columns'][0]
	logger.info(f'According to the first pkl file: target_column = {target_column}, pred_col = {pred_col}')

	#### Put the predictions from each input predictor into one DataFrame
	# Starting with the groundtruth:
	if ('sampleId' in first_pkl['df'].columns):
		new_df = first_pkl['df'][['sampleId', target_column, 'set']].copy()
		new_df.set_index('sampleId', inplace=True)
	else:
		assert first_pkl['df'].index.name == 'sampleId'
		new_df = first_pkl['df'][[target_column, 'set']].copy()

	assert new_df.index.name == 'sampleId'

	# Add each predictor's predictions to the df
	pred_columns = []
	for ob in input_dirs:
		if (ob['pkl_data']['df'].index.name != 'sampleId'):
			# ob['pkl_data']['df'].reset_index(inplace=True)
			ob['pkl_data']['df'].set_index('sampleId', inplace=True)

		assert (new_df.index.sort_values() == ob['pkl_data']['df'].index.sort_values()).all()
			
		# assert (new_df['sampleId'] == ob['pkl_data']['df']['sampleId']).all()
		# assert (new_df[target_column] == ob['pkl_data']['df'][target_column]).all()
		# assert (new_df['set'] == ob['pkl_data']['df']['set']).all()

		# save the data into the new dataframe
		nickname = ob['nickname']
		new_col = f'pred_from_{nickname}'
		# new_df[new_col] = ob['pkl_data']['df'][pred_col]
		new_df = new_df.join(ob['pkl_data']['df'][pred_col].rename(new_col))

		pred_columns.append(new_col)

	#### Build model
	df_for_training = new_df[new_df['set'].isin(SETS_FOR_TRAINING)]
	X = df_for_training[pred_columns]
	Y = df_for_training[target_column]
	logger.info(f'Using {SETS_FOR_TRAINING} sets for training')
	logger.info(f'X.shape: {X.shape}')
	logger.info(f'Y.shape: {X.shape}')
	model = linear_model.LinearRegression(fit_intercept=True, positive=WHICH_MODEL=='NNLS')
	model.fit(X.values, Y.values)

	# make predictions from the OLS model:
	X = new_df[pred_columns].values
	pred = model.predict(X)

	# Cap negative predictions to 0:
	if (CAP_NEGATIVE_PREDICTIONS):
		pred[pred<0] = 0

	# Make new dataframe to hold groundtruth and the new predictions
	groundtruth_df = new_df[[target_column, 'set']].copy()
	new_pred_column = f'{target_column}_pred'
	pred_df = pd.DataFrame(pred, columns=[new_pred_column], index=groundtruth_df.index)
	final_df = groundtruth_df.join(pred_df)

	# calc RMSE
	result_info = OrderedDict()
	result_info['kth_fold'] = kth_fold
	for which_set, group_df in final_df.groupby('set'):
		score = rmse(group_df[target_column], group_df[new_pred_column])
		result_info[f'RMSE_{which_set}'] = score
		
	# and store weights
	result_info['intercept_and_weights'] = [model.intercept_] + model.coef_.tolist()
	logger.info(json.dumps(result_info, indent=2))

	# save human-readable result info
	with open(join(out_dir, 'result_info.json'), 'w') as text_file:
		text_file.write(json.dumps(result_info, indent=2))

	# save predictions to pkl
	logger.info('Saving best_model_predictions.pkl...')
	predictions_dict = {
		'df': final_df,
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

	# plot weights
	fig, ax = plt.subplots(1, 1, figsize=(10,5))
	xt = range(len(pred_columns)+1)
	ax.bar(xt, [model.intercept_] + model.coef_.tolist())
	ax.set_xticks(xt)
	ax.set_xticklabels(['Intercept'] + pred_columns);
	ax.set_ylabel('Weight');
	fig.tight_layout()
	fig.savefig(join(out_dir, 'weights.png'))

	return predictions_dict, result_info

def main(target, m, prefix='', input_dirs=None, resultsDir=None):
	"""
	Args:
		target: name of target variable (Li, Zr, etc)
		m: comment_message used to name the output task directory
		prefix: prefix to put before the model name (e.g. use "PLS+" for high-level fusion that utilizes PLS predictions)
		input_dirs: list of dict describing where the predictions are saved that will be used as input to this model.
		  e.g. [{'dir_path':r'C:/PLS/Li_sensor=SNV_XRF,n_comp=5','nickname':'XRF'},{'dir_path':r'C:/PLS/Li_sensor=SNV_XRF,n_comp=5','nickname':'HYPER'},{'dir_path':r'C:/PLS/Li_sensor=SNV_LIBS,n_comp=20','nickname':'LIBS'},]
		resultsDir: where to put output directory
	"""

	############################# SETTING UP ###############################
	# get configuration settings
	configData, _ = getConfigOrCopyDefaultFile(CONFIG_FPATH, CONFIG_DEFAULT_FPATH)
	resultsDir = get_results_dir(configData, resultsDir)

	########################################################################
	### Get existing predictions from already-trained models (i.e. predictors) 
	if (input_dirs is None):
		print('No `input_dirs` specified on command line; entering interactive mode...')
		input_dirs = []
		while True:
			answer = input('Enter path to directory containing model training results (or blank when done): ')
			if (answer == ''):
				print(f'{len(input_dirs)} directories have been entered')
				break
			elif (exists(answer)):
				input_dirs.append({
					'dir_path': answer,
					'nickname': input('Enter nickname for this predictor: '),
				})
			else:
				print('Filepath does not exist')

	### Make output directory
	out_dir, task_run_id = prepare_out_dir_and_logging(
		resultsDir=resultsDir, 
		comment_message=m,
		cmd_args_dict={'input_dirs': input_dirs}, # log these settings
		global_id_fpath=None,
		use_task_run_id=True,
	)
	logger = logging.getLogger('spectra_ml')
				
	### How many folds are there?
	tmp_path = input_dirs[0]['dir_path']
	num_folds = len([x for x in os.listdir(tmp_path) if x.startswith('k=')])
	logger.info(f'Detected {num_folds} folds')

	### Loop over each fold...
	result_infos = []
	for kth_fold in range(num_folds):
		logger.info(_c.m_green(f'\n===RUNNING fold {kth_fold} ==='))

		subdir = join(out_dir, f'k={kth_fold}')
		os.makedirs(subdir)

		predictions_dict, result_info = process_each_fold(subdir, kth_fold, input_dirs, configData)

		# save for later
		result_infos.append(result_info)

	### Save in format that the evaluation plots use:
	assert predictions_dict['target_columns'][0][:2] == target
	ser = pd.Series({
		'target': target,
		'result_info_per_fold': pd.DataFrame(result_infos),
		'SOURCE_OF_THIS_FILE': 'train_OLS_sensor_fusion.py',
		'date_created': str(dt.now()),
		'task_dir_path': out_dir,
		'WHICH_MODEL': WHICH_MODEL,
		'input_dirs': input_dirs,
	})

	nicknames = [ob['nickname'] for ob in input_dirs]
	# out_fname = f"{prefix}{WHICH_MODEL},{target},{'_'.join(nicknames)}"
	out_fname = f"{prefix}{WHICH_MODEL},{target}"
	ser.to_pickle(join(out_dir, out_fname + '.pkl'))
	ser.to_json(join(out_dir, out_fname + '.json'), indent=4)



if __name__ == '__main__':
	import fire
	fire.Fire(main)