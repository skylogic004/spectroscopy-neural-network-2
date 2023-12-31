from hyperopt import hp
from hyperopt.pyll.base import scope
import math
from copy import deepcopy
from pprint import pprint, pformat

__author__ = "Matthew Dirks"

### HELPER CONSTANTS
WHOLE_NUMBERS = 1
 
# round the input to n significant figures
round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
round_to_3 = lambda x: round_to_n(x, 3)

def hp_idx(name, options):
	# an ordinal random integer is used to index into a list of options
	assert name.endswith('_idx')

	return hp_ordinal_randint(name, 0, len(options)-1)

def hp_nominal_randint(name, a, b):
	""" hp.randint stupidly returns a float. This function fixes that. 
	NOTE! these numbers ARE NOT CORRELATED; hyperopt does not consider, say, 1 to be closer to 2 than to 9. 
	i.e. these are nominal values NOT ordinal.

	IN HINDSIGHT: hp.uniformint might be a built-in better version of randint

	In the source code:
		> randint can be seen as a categorical with high - low categories
		(tpe.py:575)

	Args:
		a: min value
		b: range is up to, but not including, b
	"""
	return scope.int(hp.randint(name, a, b))

def hp_ordinal_randint(name, low, high):
	return hp_better_quniform(name, low, high, WHOLE_NUMBERS)

def hp_better_quniform(name, low, high, q):
	""" hp.quniform stupidly ISN'T UNIFORM. The low and high numbers get half the probability they should (probability is half of what all the numbers >low and <high get).
	This function fixes that.
	Also, quniform does weird rounding things such that sometimes a number is returned from outside the [low, high] range. 
	This fixes that too.
	See https://github.com/hyperopt/hyperopt/issues/328

	Also, this function casts as int as long as the `q` is also an int.

	Note: hyperopt will internally store values as their index in the database. e.g. quniform('foo', 5, 10, 1) will store x in [0, 4] rather than in [5, 10]
	"""
	_range = high - low
	assert _range % q == 0, f'You would get samples greater than high ({high}) if range (high - low) does not factor into q ({q}) nicely, so I dont allow this'

	var = hp.quniform(name, -q/2, high-low+q/2, q) + low
	if (isinstance(q, int)):
		return scope.int(var)
	else:
		return var


def process_sample(sample_of_cmd_space):
	assert isinstance(sample_of_cmd_space, dict)

	d = deepcopy(sample_of_cmd_space)

	# process the parent dict
	if ('$process_function' in d):
		d['$process_function'](d)

	# process these sub-dicts
	for key in ['enc_options', 'dec_options']:
		if (key in d):
			if ('$process_function' in d[key]):
				d[key]['$process_function'](d)

	return d


def my_pformat(a_dict, width=200):
	""" pformat but hide some things for readability.
	`width` of 200 is good for printing on-screen in Jupyter notebook,
	but more like 500 is good for saving to file.
	"""

	is_iter = lambda x: isinstance(x, tuple) or isinstance(x, list)

	def _helper(d):
		if (isinstance(d, dict)):
			for key, value in d.items():
				if (key == 'kernel_init' and is_iter(value)):
					d[key] = '[HIDDEN_FOR_READABILITY]'
				elif (isinstance(value, dict)):
					d[key] = _helper(value)
				elif (is_iter(value)): # is it iterable?
					value = list(value) # make sure it's mutatable
					for idx, entry in enumerate(value):
						value[idx] = _helper(entry)

		return d

	new_dict = _helper(a_dict)
	return pformat(new_dict, width=width)

def my_pprint(a_dict, width=200):
	print(my_pformat(a_dict, width))

def make_odd(x):
	""" Make x odd by adding 1 if the x is even. """
	x = int(x)
	if ((x % 2) == 0):
		return x + 1
	else:
		return x