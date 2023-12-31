import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # ignore "RuntimeWarning: invalid value encountered in true_divide"
import os
import time
from os.path import join, split, realpath, dirname, exists
import logging
from sklearn import metrics
import pickle
from tqdm import tqdm
import math
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict

from spectra_ml.components.plot.subplotShape import makeNiceGridFromNElements
from spectra_ml.components.plot import coloredScatter
from spectra_ml.components.scores import score

__author__ = "Matthew Dirks"

COLORS = {
	'train': '#000000',
	'dev': '#365CFF',
	'test': '#FF0000',
	'test_CV': '#FF9000',
	'train_or_dev': '#001A87',
}

# setup colors and marker styles
SCATTER_STYLES = {key:{'c':c,'marker':'o','lw':0} for key, c in COLORS.items()}
for _set in ['dev', 'test']:
	SCATTER_STYLES[_set]['marker'] = 'x'
	SCATTER_STYLES[_set]['lw'] = 1
	SCATTER_STYLES[_set]['alpha'] = 1.0
SCATTER_STYLES['train']['alpha'] = 0.6
SCATTER_STYLES['train_or_dev']['alpha'] = 0.6
SCATTER_STYLES['test_CV']['marker'] = '2'
SCATTER_STYLES['test_CV']['lw'] = 1

def subplots_to_1D_list(axs):
	"""
	Args:
		axs: a single SubplotAxes or many (as returned by plt.subplots(shape))
	"""
	if (isinstance(axs, mpl.axes.Subplot)):
		axs = [axs]
	elif (len(axs.shape) > 1):
		axs = [item for sublist in axs for item in sublist]
	return axs

def subplots_to_2D_matrix(axs, n_rows, n_cols):
	# ensure that axs is a matrix always, even when some dimensions are 1
	if (n_rows == 1 and n_cols == 1):
		axs = [[axs]]
	if (n_rows == 1 and n_cols > 1):
		axs = [axs]
	if (n_rows > 1 and n_cols == 1):
		axs = [[x] for x in axs]
	axs = np.array(axs)
	return axs

def plot_target_vs_predictions(out_dir, uniqueId, joined_df, target_columns, fig_title=None):
	""" copied from core_hyperspectral_NN.plotters,
	then modified. """
	nTargets = len(target_columns)

	(nRows, nCols), _ = makeNiceGridFromNElements(nTargets)
	fig, subs = plt.subplots(nRows, nCols, figsize=(4*nCols, 4*nRows))
	subs = subplots_to_1D_list(subs)

	for sub, _target in zip(subs, sorted(target_columns)):
		_prediction_column = _target+'_pred'
		x_groundtruth = joined_df[_target].astype(float)
		y_predictions = joined_df[_prediction_column].astype(float)

		cs = joined_df['set'].apply(lambda x: COLORS[x])

		sub.scatter(x_groundtruth, y_predictions, c=cs, alpha=0.6)
		_range = [min(0, min(x_groundtruth), min(y_predictions)), max(0, max(x_groundtruth), max(y_predictions))]
		sub.plot(_range, _range, ls='--', c='#BBBBBB')
		sub.set_xlim(_range); sub.set_ylim(_range)

		sub.set_xlabel(_target)
		sub.set_ylabel('prediction')
		sub.set_aspect('equal')

	if (fig_title is not None):
		plt.figtext(0, 0, r'{}'.format(fig_title))

	fig.tight_layout()
	fig.savefig(join(out_dir, 'predictions_%s.png' % uniqueId))
	plt.close(fig)

def plot_target_vs_predictions_individual(joined_df, target_columns, out_dir):
	log = logging.getLogger('spectra_ml')
	output_st = time.time()
	nTargets = len(target_columns)

	if (not exists(out_dir)):
		os.makedirs(out_dir)

	for _target in target_columns:
		_prediction_column = _target+'_pred'
		_std_column = f'{_target}_std'

		### Loop over: train, dev, and test sets, as well as ALL sets
		subsets = list(joined_df.groupby('set')) + [('ALL', joined_df)]

		for _set, set_df in subsets:
			if (_set in ['test', 'test_CV', 'ALL']): # only plot these ones
				MAE, MSE, RMSE, R2, num_points = score(set_df, _set, _target, _prediction_column)

				title = 'MAE={mae:0.4f}, R2={r2:0.3f}, RMSE={rmse:0.4f} (set:{_set})'.format(mae=MAE, r2=R2, rmse=RMSE, _set=_set)

				# calculate axis limits
				_max = set_df[[_target, _prediction_column]].max().max() # max across both axes
				xlim = [0, set_df[_target].max()]
				ylim = [0, _max] # from 0 to max(max(prediction), max(target)), so that if predictions are lower than target, we'll keep the figure square, if predictions are greater than target, then stretch out into a rectangle shaped figure

				# special case: if the range of either axis is 0, try to copy the other axis instead
				if (xlim[1] - xlim[0] == 0):
					xlim = ylim
				if (ylim[1] - ylim[0] == 0):
					ylim = xlim

				# calculate dynamic figure size, based on axis bounds that will be set later
				w = 12 # always same width image

				# height is 8 if y-axis has same limits has x-axis. height is 16 is y-axis has twice the limit of x-axis. etc
				ratio = ylim[1] / xlim[1]
				if (ylim[1] == xlim[1]):
					# when the limit goes to 0, we get inf value here, so setting ratio to 1 prevents issues when trying to plot
					ratio = 1
				h = 8 * ratio

				# cap the height (sometimes predictions explode, causing height of figure to explode)
				h = min(h, 20)

				# make scatter plot
				fig = coloredScatter.make(set_df, _target, _prediction_column, 
					title=title,
					groupByColumn='set',
					groupStyles=SCATTER_STYLES,
					equalAspectRatio=True,
					figsize=(w,h),
				)
				ax = fig.gca()

				# plot error bars if standard deviation present
				if (_std_column in set_df):
					for _, row in set_df.iterrows():
						std = row[_std_column]
						x = row[_target]
						y = row[_prediction_column]
						ax.plot([x]*2, [y-std*2, y+std*2], c=COLORS[row['set']], lw=0.5)

				# plot ideal regression line (straight diagonal line)
				ax.plot([0,_max], [0,_max], '--', c='#AAAAAA', alpha=0.5)

				# axis limits (fixed width, expanding height)
				ax.set_xlim(xlim)
				ax.set_ylim(ylim)

				fig.tight_layout()

				# save fig
				fname_format = 'predictions_{target}_{_set}.{ext}'
				fname = fname_format.format(target=_target, _set=_set, ext='png')
				# bbox_inches and pad_inches crops the whitespace around the figure
				fig.savefig(join(out_dir, fname), bbox_inches='tight', pad_inches=0.1)
				plt.close(fig)


	log.info('TIME--plot_target_vs_predictions_individual: %0.3f s' % (time.time() - output_st))

	
def plot_history(out_fpath, history):
	fig = plt.figure(figsize=(8,4))
	plt.plot(history['loss'], label='Calibration set loss')

	if ('val_loss' in history):
		plt.plot(history['val_loss'], label='Tunning set loss')
		
	plt.yscale('log')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()

	if ('lr' in history):
		ax2 = plt.gca().twinx()
		ax2.plot(history['lr'], color='r', ls='--')
		ax2.set_ylabel('learning rate', color='r')
		ax2.set_yscale('log')
		plt.tight_layout()
		
	plt.savefig(out_fpath, dpi=96)
	plt.close()