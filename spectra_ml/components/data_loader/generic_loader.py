import pandas as pd
from sklearn.utils import shuffle
import logging
from os.path import exists, join
import pickle
from sklearn.model_selection import LeaveOneOut, LeavePOut, ShuffleSplit, KFold, RepeatedKFold
from tqdm import tqdm
from collections import OrderedDict, Counter
from sklearn.utils import resample
import numpy as np
from colorama import Fore, Back, Style

from spectra_ml.components.data_loader import mangoes

DATASETS = {
	# 'pharm_tablets': pharmaceutical_tablets.load_from_source,
	# 'COVID19': COVID19.load_from_source,
	'mangoes': mangoes.load_Anderson_data,
	# 'mangoes_Dario': mangoes.load_Dario_data,
	'Li_hole1': 'PROPRIETARY DATASET; NOT RELEASED',
}

EXPECTED_DATASET_NAMES = list(DATASETS.keys())

def load(dataset_name, data_dir):
	"""  if data has already been processed, load it directly (to save time and compute),
	otherwise process the data and save the result. """
	logger = logging.getLogger('spectra_ml')

	fname = f'PROCESSED_{dataset_name}.pkl'
	fpath = join(data_dir, fname)

	logger.info(f'Checking if processed data exists in {fpath}')

	if (exists(fpath)):
		logger.info('Reading data from {}...'.format(fpath))
		#with open(fpath, 'rb') as f:
		dataset_dict = pd.read_pickle(fpath)
	else:
		logger.info('Processed data doesn\'t exist, running data loader process now...'.format(fpath))

		# get dataset-specific loader function
		f = DATASETS[dataset_name]

		# call loader function
		dataset_dict = f()

		with open(fpath, 'wb') as f:
			pickle.dump(dataset_dict, f)

		logger.info('Data read from source (and written to {}).'.format(fname))

	return dataset_dict
