import os, sys
from os.path import join, split, realpath, dirname, splitext, isfile, exists, isdir
import logging
import json
import pprint
import shutil
import random
import subprocess
import time
import traceback
import posixpath

from spectra_ml.components.make_output_dir import newDir
from spectra_ml.components.console_logging import setup_logging, end_logging

__author__ = "Matthew Dirks"

IS_LINUX = os.name == 'posix'

def prepare_out_dir_and_logging(#SYMLINK_FPATH=None, 
								resultsDir=None, 
								comment_message=None, 
								cmd_args_dict=None, 
								global_id_fpath=None,
								fallback_resultsDir=None,
								use_task_run_id=True):

	# prepare to save outputs
	if (comment_message is None):
		comment_message = input('Enter message for output directory name: ')
		comment_message = comment_message if (len(comment_message)>0) else None

	#########################################
	##### Make output directory (outDir)
	#########################################
	task_run_id = None

	if (use_task_run_id):
		# look around for the global_id file
		paths_to_try = []
		if (global_id_fpath is not None):
			paths_to_try.append(global_id_fpath)

		paths_to_try.extend([
			join(resultsDir, 'global_id'),
			join(resultsDir, '..', 'global_id'),
		])

		if (fallback_resultsDir is not None):
			paths_to_try.append(join(fallback_resultsDir, 'global_id'))

		found_global_id_fpath = None
		for fp in paths_to_try:
			if (exists(fp) and isfile(fp)):
				found_global_id_fpath = fp
				break
			else:
				print('Looking for global id file... this one doesn\'t exist: {}'.format(fp))

		if (found_global_id_fpath is None):
			raise(Exception("Cannot find a global_id file. If you haven't made one yet, you should create an empty text file called \"global_id\" in the same directory as `results_dir` (which you specify in `config.toml`), then write \"0\" in the file (or any number) which is the starting ID to use in naming output directories."))

		# create output directory
		outDir, task_run_id = newDir(
			realpath(resultsDir), 
			name=comment_message, 
			id_fpath=found_global_id_fpath,
			# taskKey=task_key,
		)
	else:
		outDirName = comment_message
		outDir = posixpath.join(realpath(resultsDir), outDirName)
		os.makedirs(outDir)


	#########################################
	##### Save stuff into the outDir that has now been created
	#########################################

	# save command line used to run (simple text file)
	with open(join(outDir, 'cmd.txt'), 'w') as text_file:
		text_file.write(' '.join(sys.argv))

	# save cmd line args (nicely formatted)
	if (cmd_args_dict is not None):
		# with open(join(outDir, 'cmd_args.json'), 'w') as f:
			# json.dump(cmd_args_dict, f, indent=2)

		with open(join(outDir, 'cmd_args.pyon'), 'w') as f:
			txt = pprint.pformat(cmd_args_dict, width=100)
			f.write(txt)

	# if an input file was used to specify the command line args, copy that file too
	if ('cmd_args_fpath' in cmd_args_dict and cmd_args_dict['cmd_args_fpath'] is not None):
		fname = split(cmd_args_dict['cmd_args_fpath'])[1]
		shutil.copy(cmd_args_dict['cmd_args_fpath'], join(outDir, 'cmd_args_base_template.pyon'))

	# save console output to log
	logger = setup_logging(outDir)

	return outDir, task_run_id
