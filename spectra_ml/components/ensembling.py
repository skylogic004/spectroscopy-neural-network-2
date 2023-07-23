import pandas as pd
from collections import OrderedDict
import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from os.path import join
import pickle

__author__ = "Matthew Dirks"

rmse = lambda arr1, arr2: np.sqrt(mean_squared_error(arr1, arr2))

def flatten(l):
	""" Takes a list of lists and returns a single list containing all the elements.
	e.g. [[1,2],[3,4]] becomes [1,2,3,4]
	"""
	flat_list = [item for sublist in l for item in sublist]
	return flat_list

class Ensembler:
	def __init__(self):
		self.metadata_of_models = OrderedDict()

	def record_metadata_for_run(self, ith_run, run_result):
		self.metadata_of_models[ith_run] = run_result

	def get_epochs(self):
		""" This info used to plot events on loss history plot """
		return [model_metadata['epoch'] for model_metadata in self.metadata_of_models.values()]

	def make_final(self, target_columns_in_use):
		""" compute mean predictions for each instance from the collection (ensemble)
		of models' predictions. And return data for saving and plotting. """
		logger = logging.getLogger('spectra_ml')

		n_runs = len(self.metadata_of_models)

		logger.info(f'Ensembler collected {n_runs} models. Computing the ensemble now...')

		# collect ensemble data
		if (n_runs == 0):
			# no models were saved! This is bad.
			raise(Exception('Ensembler has 0 models to work with. Cannot proceed.'))

		# every run picks a random dev set. Will combine train and dev together
		sets_to_evaluate = ['test','test_CV','train','dev']
		first_predictions_dict = self.metadata_of_models[0]['predictions_dict']
		assert all([which_set in first_predictions_dict.keys() for which_set in sets_to_evaluate])

		# get data from first run...
		# (and combine across sets)
		# (just the 'set' and target columns)
		df = pd.concat([first_predictions_dict[which_set] for which_set in sets_to_evaluate], axis=0)[['set']+target_columns_in_use]
		assert df.index.name == 'sampleId'

		# rename "train" and "dev" to "train_or_dev"
		df.loc[df['set'].isin(['train', 'dev']), 'set'] = 'train_or_dev'

		# then get the predictions from each run
		for run_idx, metadata in self.metadata_of_models.items():
			# combine data across sets
			df2 = pd.concat([metadata['predictions_dict'][which_set] for which_set in sets_to_evaluate], axis=0)

			# rename "train" and "dev" to "train_or_dev"
			df2.loc[df2['set'].isin(['train', 'dev']), 'set'] = 'train_or_dev'

			# sanity check: making sure the data joins properly
			assert (df[['set']+target_columns_in_use].sort_index() == df2[['set']+target_columns_in_use].sort_index()).all().all()

			# add the run_idx to the column name (existing column name is in "TARGET_pred" format)
			old_column_names = [f'{target}_pred' for target in target_columns_in_use]
			new_column_names = [f'run{run_idx}:{target}_pred' for target in target_columns_in_use]
			df2.rename(columns=dict(zip(old_column_names, new_column_names)), inplace=True)

			# add columns to existing df
			df = df.join(df2[new_column_names])

		# calculate ensemble's predictions (the average) and other stats for each target
		stats_df = df[['set']+target_columns_in_use].copy()
		for target in target_columns_in_use:
			# get predictions for target
			per_run_pred_columns = [f'run{run_idx}:{target}_pred' for run_idx in self.metadata_of_models.keys()]
			predictions = df[per_run_pred_columns]

			# calc stats
			stats_df[f'{target}_pred'] = predictions.mean(axis=1)
			stats_df[f'{target}_min'] = predictions.min(axis=1)
			stats_df[f'{target}_max'] = predictions.max(axis=1)
			stats_df[f'{target}_std'] = predictions.std(axis=1)

		to_save = {
			'df': stats_df,
			'target_columns': target_columns_in_use,

			# NOTE: plotting functions hardcoded to use f'{target}_pred' format. If you change it here, plotting functions will need to change too
			'prediction_columns': [f'{target}_pred' for target in target_columns_in_use],

			# also save individual predictions from each model of the ensemble - for easy post-analysis
			'ensemble_runs_df': df,
		}

		return to_save


	def get_ensemble_RMSE(self, _set, target):
		""" Get the ensemble's score for _set and target
		(the ensemble is built using training runs completed thus far)
		"""

		# get predictions
		df = self.make_final([target])['df']

		# calc prediction accuracy
		set_df = df[df['set']==_set]
		RMSE = rmse(set_df[target], set_df[f'{target}_pred'])

		return RMSE

	def trigger_short_circuit(self, ensemble_short_circuit):
		""" Short-circuit training based on criteria supplied, if any.
		
		Returns:
			True if training should abort.
		"""
		if (ensemble_short_circuit is not None):
			logger = logging.getLogger('spectra_ml')

			n_runs_completed = len(self.metadata_of_models)

			for settings_dict in ensemble_short_circuit:
				# e.g. settings_dict may be {'at_n_runs': 10, 'RMSE_needed': 0.8, 'target': 'DM', 'set': 'dev'}

				# Check if required number of runs have completed at this point
				if (settings_dict['at_n_runs'] == n_runs_completed):
					logger.debug(f'DEBUG INFO FOR SHORT CIRCUIT: at_n_runs = n_runs_completed = {n_runs_completed}')

					# Check if current ensemble RMSE meets requirement
					RMSE = self.get_ensemble_RMSE(settings_dict['set'], settings_dict['target'])
					logger.debug(f'DEBUG INFO FOR SHORT CIRCUIT: RMSE={RMSE}, RMSE_needed={settings_dict["RMSE_needed"]}')
					if (RMSE > settings_dict['RMSE_needed']):
						# requirement not satisfied - abort
						return True


		return False
