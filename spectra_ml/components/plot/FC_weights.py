from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

from spectra_ml.components.plot.subplotShape import subplots_to_2D_matrix

__author__ = "Matthew Dirks"

def plot_FC_weights(model_cnn):
	"""
	Args:
		model_cnn: A Model instance from keras
	"""

	# get dense (FC) layer
	dense_layers = [layer for layer in model_cnn.layers if 'dense' in layer.name]
	n_layers = len(dense_layers)

	n_rows = 2
	fig, axs = plt.subplots(n_rows, n_layers, figsize=(5*n_layers, 10))
	axs = subplots_to_2D_matrix(axs, n_rows, n_layers)

	for layer_idx, (axs_col, layer) in enumerate(zip(axs.T, dense_layers)):
		weights, biases = layer.get_weights()

		### weights
		ax = axs_col[0]
		ax.set_title(layer.name)

		# the smaller dimension will looped over, producing a line for the other dimension
		smallest_dim = np.argmin(weights.shape)

		if (smallest_dim == 1):
			# swap dimensions (transpose) so that the first dimension is the smallest one always
			ax.set_xlabel(f'in-neuron idx (of {weights.shape[0]}) ({weights.shape[1]} out-neurons = lines)')
			weights = weights.T

		else:
			ax.set_xlabel('out neuron idx')
			ax.set_xlabel(f'out-neuron idx (of {weights.shape[1]}) ({weights.shape[0]} in-neurons = lines)')

		ax.set_ylabel('weight')

		# now plot a line for every row of `weights`
		for row in weights:
			ax.plot(row, alpha=0.6)

		# make font smaller
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
			item.set_fontsize(8)

		# show min/max values via on-screen text
		# plt.figtext(layer_idx/n_layers, 0, r'{:.1E}$\rightarrow${:.1E}'.format(weights.min(), weights.max()))

		### biases
		ax2 = axs_col[1]
		ax2.set_title(f'{layer.name} bias')
		ax2.bar(range(len(biases)), biases)

	fig.tight_layout()

	return fig