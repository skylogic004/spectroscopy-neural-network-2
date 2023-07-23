"""
This sets up the hyperparameter space for hyperopt to search over; 
a sample of this space is converted into a "cmd" (i.e. the command line arguments for the main python script).
`HPO_start_master.py` calls the `get_cmd_space` specified here.
"""

__author__ = "Matthew Dirks"

def get_cmd_space(which_cmd_space, target):
	""" This function declares the space of hyperparameters to optimize over.
	Since these parameters are ultimately command-line arguments, I call it "cmd_space".
	"""
	found = False

	from spectra_ml.components.hyperopt_config import cmd_space_for_mangoes, cmd_space_for_mangoes_sensor_fusion
	modules_to_search = [cmd_space_for_mangoes, cmd_space_for_mangoes_sensor_fusion]

	for module in modules_to_search:
		SPACE_NAMES = list(module.SPACE_VARIANTS.keys())

		if (which_cmd_space in SPACE_NAMES):
			cmd_space, hyperhyperparams = module.get_cmd_space(which_cmd_space, target)
			found = True
			break

	if (not found):
		raise(ValueError('Invalid cmd_space name.'))

	return cmd_space, hyperhyperparams
