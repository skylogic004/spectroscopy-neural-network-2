import fire
import pandas as pd
from collections import OrderedDict
import itertools
import numpy as np
import os
from os.path import exists, isdir
from posixpath import join, normpath
import json
import colorama
from colorama import Fore, Back, Style
import types
import pprint
import json

__author__ = "Matthew Dirks"

def dict_to_cmd_arg(ob):
	# format as JSON, no newlines and no indentation
	arg_str = json.dumps(ob, indent=None)

	# windows command line requires args to be quoted in double quotes, and therefore the inner quotes need to single quotes
	arg_str = arg_str.replace('\"', '\'')

	# wrap in double quotes
	arg_str = '"{}"'.format(arg_str)

	# the keyword 'None' is a python literal not a string (json wraps it in quotes)
	arg_str = arg_str.replace('\'None\'', 'None')

	# json converts None into 'null' (None literal to string 'null') (note: the above line may be obsolete, b/c None's don't seem to pass through)
	# convert back to None without quotes (python literal)
	arg_str = arg_str.replace('null', 'None')

	# json uses true/false as literals, but I want the Python literals for booleans
	arg_str = arg_str.replace('true', 'True')
	arg_str = arg_str.replace('false', 'False')

	return arg_str

def build_cmd(experiment_name, row):
	if ('m' not in row): # message *not* manually specified
		message = 'cmd{}'.format(row['cmd_num'])
	else:
		# override: just use the given message in `m`
		message = row['m']

	assert 'model_name' not in row

	if ('n_training_runs' in row):
		python_program = 'train_model.py NN'

	cmd = 'python {python_program}'.format(python_program=python_program)

	cmd += ' --m "{}"'.format(message)

	# all key-value-pairs in row are added as keyword arguments to the command line
	# by looping through them all here (and applying some conversions as needed)
	for key in row.index:
		try:
			value = row[key]

			### Dealing with data types ###
			if (value is None):
				arg_value = 'None'

			elif (key in ['conv_filter_width', 'conv_n_filters']):
				if (isinstance(value, int)):
					arg_value = str(value)
				elif (isinstance(value, tuple) or isinstance(value, list)):
					arg_value = f'"{value}"'
				else:
					raise(ValueError(f'Invalid type for cmd arg {key} which has type {type(value)}'))

			# explicit surrounding quotes (strings, or arrays of primitives)
			elif (key in ['cmd_args_fpath', 'resultsDir']):
				arg_value = '"{}"'.format(value)

			# whatever values (string, float, bool, int, None)
			elif (key in ['model_name', 'LR', 'out_dir_naming', 
			              'FC_L2_reg_factor', 'conv_filter_init', 'conv_L2_reg_factor', 'FC_init',
			              'do_LR_finder', 'do_ES', 'dataset_name', 'which_targets_set', 'base_LR', 'base_min_LR',
			              'conv_proximity_L2_factor', 'BN',
			              ]):
				arg_value = '{}'.format(value)

			# float values
			elif (key in []):
				arg_value = '{:f}'.format(value)

			# integer values
			elif (key in ['kth_fold', 'n_epochs', 'n_gpu', 'n_cpu', 'batch_size',
			              'n_full_epochs', 'n_in_parallel', 'seed_start', 'n_training_runs',
			              'ES_patience']):
				arg_value = '{:d}'.format(int(value))

			# array values, where each array contains any python literal (dict, tuple, str, etc)
			elif (key in ['input_features', 'ensemble_short_circuit', 'FC_size_per_input', 'drop_LR_at_epochs', 
			              'which_folds', 'FC_L2_reg_factor_per_input']):
				arg_value = '"{}"'.format(str(value))

			# array values, where each array contains only strings
			elif (key in ['ignore_samples']):
				array_as_string = ','.join(['\'{}\''.format(x) for x in value])
				arg_value = '"[{}]"'.format(array_as_string)

			# dict values
			elif (key in ['fold_spec', 'model_params', 'LR_sched_settings', 'loss_func', 'scaler_settings']):
				arg_value = '{:s}'.format(dict_to_cmd_arg(value))

			# array of ints
			elif (key in ['kth_folds']):
				assert isinstance(value, list), '{} must be a list'.format(key)
				arg_value = '[{}]'.format(','.join([str(x) for x in value]))

			# ignore these
			elif (key in ['cmd_num', 
			              'm', # 'm' is handled separately above, so ignore it here.
			             ]):
				arg_value = None
				
			# ignore these
			elif (key.startswith('_')):
				arg_value = None

			else:
				raise(ValueError('make_cmd: Couldn\'t figure out what to do with key ({})'.format(key)))
		except Exception as e:
			print(f'DEBUGGING INFO: key={key}, value={value}, value is None? {value is None}, value type={type(value)}')
			raise(e)

		# append key and converted value (arg_value) to cmd string
		if (arg_value is not None):
			cmd += ' --{key} {arg_value:s}'.format(key=key, arg_value=arg_value)

	return cmd
