import os, sys, argparse

import sys
sys.path.append('../code')
from classes.ep_sampler import ExtendedProteinSampler
from utils.general import load_inputs, read_from_name, create_path, find_path


# Load the input files, check them and
# and make the results directory
def prepare_directory(args):
	pars = {key: load_inputs(args.pars_file, start=f"## {key}", end="##") for key in ["protein", "sampler"]}
	settings = load_inputs(args.settings_file)
	config_settings = load_inputs(args.cs_file)

	pars["protein"]["ref_seq"], _ = read_from_name(pars["protein"]["ref_code"], pars["protein"]["pdb_dir"], return_attmatrix=False)

	create_path(settings['results_dir'])
	settings['results_dir'] = find_path(raw_path=settings['results_dir'], dname='sim', pfile=args.pars_file, pname='pars.txt', lpfunc=load_inputs)
	settings['eprot_dir'] = f"{settings['results_dir']}/eprot"
	create_path(settings['eprot_dir'])

	return pars, settings, config_settings


# Main
def main(args):
	print(f'PID: {os.getpid()}\n')

	print('Loading inputs...')
	pars, settings, config_settings = prepare_directory(args)

	print('Initializing sampler...')
	sampler = ExtendedProteinSampler(config_settings=config_settings)

	print(f'Starting the simulation!')
	sampler.sample(
			ref_seq=pars['protein']['ref_seq'],
			pars=pars['sampler'],
			settings=settings,
	)
	print(f'\nSimulation completed!')
    

def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--pars-file",
		type = str,
		default = "inputs/pars.txt",
		help = "str variable, path to the parameters file used in the simulation. Default: 'inputs/pars.txt'."
	)
	parser.add_argument(
		"--settings-file",
		type = str,
		default = "inputs/settings.txt",
		help = "str variable, path to a secondary input file for specifics which do not alter the simulation. Default: 'inputs/settings.txt'."
	)
	parser.add_argument(
		"--cs-file",
		type = str,
		default = "../CONSTANTS.txt",
		help = "str variable, path to the common parameters file used for structure prediction (if implemented). Default: '../CONSTANTS.txt'."
	)
	return parser
    
if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	main(args)
