from tabulate import tabulate
from os.path import join, exists
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from textwrap import indent
from collections import OrderedDict
import pickle
import json

from spectra_ml.components.spectroscopy_NN import plotters
from spectra_ml.components.ensembling import Ensembler, rmse

__author__ = "Matthew Dirks"

def indent_and_log(txt):
	logger = logging.getLogger('spectra_ml')
	logger.info(indent(txt, ' '*4))

def get_run_out_dir(ith_run, out_dir):
	return join(out_dir, f'run_{ith_run:02d}')

def finish_ensemble(out_dir, configData, ensembler, target_columns_in_use, kth_fold):
	logger = logging.getLogger('spectra_ml')

	assert len(target_columns_in_use)==1, 'This code written for one target only. TODO: support multiple.'
	target = target_columns_in_use[0]

	# process results into ensemble
	to_save = ensembler.make_final(target_columns_in_use)

	# save predictions to pkl
	logger.info('Saving best_model_predictions.pkl...')
	to_save['kth_fold'] = kth_fold
	with open(join(out_dir, 'best_model_predictions.pkl'), 'wb') as f:
		pickle.dump(to_save, f)

	# plot predictions
	if (configData['plot']['target_vs_predictions']):
		logger.info('Plotting final version of targets vs predictions (individual plots)')

		plotters.plot_target_vs_predictions_individual(
			to_save['df'],
			target_columns_in_use,
			join(out_dir, 'prediction_plots'), 
		)

	# save human-readable result info
	result_info = OrderedDict()

	try:
		ens_df = to_save['df']
		for which_set, group_df in ens_df.groupby('set'):
			result_info[f'ensemble_RMSE_{which_set}'] = rmse(group_df[target], group_df[f'{target}_pred'])
	except Exception as e:
		print(f'FAILED TO RECORD RMSE scores ({e})')

	# number of epochs used in each training run
	try:
		num_epochs_per_run = []
		for run_result in ensembler.metadata_of_models.values():
			num_epochs_per_run.append(len(run_result["h.history"]["loss"]))
		result_info['num_epochs_per_run'] = num_epochs_per_run
	except Exception as e:
		print(f'FAILED TO RECORD num_epochs_per_run ({e})')


	with open(join(out_dir, 'result_info.json'), 'w') as text_file:
		text_file.write(json.dumps(result_info, indent=2))

	return to_save

def finish_training_run(out_dir, configData, ith_run, run_result, target_columns_in_use):
	logger = logging.getLogger('spectra_ml')
	run_out_dir = get_run_out_dir(ith_run, out_dir)

	# make output dir only if needed
	out_dir_needed = configData['plot']['loss_history'] or configData['plot']['target_vs_predictions_per_run']
	if (out_dir_needed and not exists(run_out_dir)):
		os.makedirs(run_out_dir)

	# print info
	indent_and_log(f'Time for this run: {run_result["time_spent"]} (hh:mm:ss.micro)')
	indent_and_log(f'Number of epochs completed: {len(run_result["h.history"]["loss"])}')
	indent_and_log('Scores: ' + str(run_result['scores']))

	# print DEBUG info
	indent_and_log(run_result['DEBUG'])

	# plot loss history
	if (configData['plot']['loss_history']):
		plotters.plot_history(join(run_out_dir, 'loss_history.png'), run_result['h.history'])

	# scatter plot (if enabled)
	if (configData['plot']['target_vs_predictions_per_run']):
		df = pd.concat(list(run_result['predictions_dict'].values()), axis=0)

		plotters.plot_target_vs_predictions_individual(
			df,
			target_columns_in_use,
			run_out_dir,
		)


	# save these figures, if they exist
	figure_names = ['LR_finder_fig', 'LR_finder_fig_2', 'scatter_fig', 'conv_filters_fig', 'FC_weights_fig']
	for fig_name in figure_names:
		if (fig_name in run_result):
			fig = run_result[fig_name]
			fig.savefig(join(run_out_dir, f'{fig_name.replace("_fig","")}.png'))
			plt.close(fig)
	if ('LR_finder_info' in run_result):
		with open(join(run_out_dir, 'LR_finder_info.json'), 'w') as f:
			json.dump(run_result['LR_finder_info'], f)
