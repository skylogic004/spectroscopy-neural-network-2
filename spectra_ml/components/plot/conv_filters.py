from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

from spectra_ml.components.plot.subplotShape import subplots_to_2D_matrix

__author__ = "Matthew Dirks"

def plot_conv_filters(model_cnn):
	"""
	Args:
		model_cnn: A Model instance from keras
	"""

	# plot conv filters
	conv_layers = [layer for layer in model_cnn.layers if 'conv' in layer.name]
	n_conv_layers = len(conv_layers)

	max_n_filters = max([layer.get_weights()[0].shape[2] for layer in conv_layers])
	fig, axs = plt.subplots(n_conv_layers, max_n_filters, figsize=(max_n_filters*4, n_conv_layers*2))

	# ensure that axs is a matrix always, even when some dimensions are 1
	axs = subplots_to_2D_matrix(axs, n_conv_layers, max_n_filters)

	for layer, axs_row in zip(conv_layers, axs):
		filters, biases = layer.get_weights()
		nx, ny, n_filters = filters.shape

		for filter_idx in range(n_filters):
			ax = axs_row[filter_idx]

			if (nx == 1 and ny == 1): # scalar (1x1 conv)
				_filter = filters[0, 0, filter_idx]
				ax.plot([0]*2, [0, _filter], c='#A80000', label='kernel')

			if (ny == 1): # 1D filter
				_filter = filters[:, 0, filter_idx]
				ax.plot(_filter, c='#A80000', label='kernel')

				ax.text(0.5, 0.05, 'bias={:0.2E}'.format(biases[filter_idx]), 
					verticalalignment='bottom', horizontalalignment='center', 
					transform=ax.transAxes, 
					fontsize=8)#, color='green')

			else: # 2D filter
				_filter = filters[:, :, filter_idx]
				ax.imshow(_filter, cmap='hot', interpolation=None)
				ax.yaxis.set_major_locator(MaxNLocator(integer=True))


			title = f'{layer.name} filter {filter_idx}'
			ax.set_title(title)

			ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	fig.tight_layout()
	return fig

