import os
from io import StringIO
from time import process_time as ptime

import torch
import math

# CUSTOM ESM
#from huggingface_hub import login
from custom_esm.models.esm3 import ESM3
from custom_esm.sdk.api import ESM3InferenceClient

# SAMPLING
import sys
sys.path.append('..')
from .ExtendedProtein import ExtendedProtein
from generator.custom_generator import CustomGenerator
from utils.general import create_path
from utils.predictions import init_structure_config      #, predict_structure          | (for now, no contact
from utils.energies import compute_U_structure_ce, compute_entropy #, compute_U_cm, compute_FoC  |  maps are computed)
from utils.operations import compute_Hd, mutate, is_subset, merge_dict



# ------------------------------------------ #
# Hybrid Monte Carlo ExtendedProtein Sampler #
# ------------------------------------------ #
class ExtendedProteinSampler():

	def __init__(
			self,
			config_settings: dict,
			name: str = 'ExtendedProteinSampler',
	):
		self.config_settings = config_settings
		self.config = init_structure_config(**self.config_settings)
		self.name = name

		# The next commented line is necessary only for the first login
		#login()
		self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open")

		self.model.eval()	#setta il modello in modalità di valutazione (disabilita dropout e batch normalization)

		for param in self.model.parameters():
			param.requires_grad = False

		self._init_attributes()



	def sample(
			self,
			ref_seq: str,
			pars: dict,
			settings: dict | None = None,
	):
		eprot, data, pars, settings = self._setup(ref_seq, pars, settings)

		# Save starting ExtendedProtein
		eprot_i = eprot.copy()
		accepted_total = data["acc_moves"]

		for move in range(data["move"]+1, pars["moves"]+1):

			# Extract momenta and integrate the equations of motion
			eprot, dU_int, dK = self._extract_and_integrate(eprot, pars)

			# Collapse and compute energy difference w.r.t. starting ExtendedProtein
			eprot.collapse()

			# Check whether the integration produced a new discrete sequence
			Hd_move = compute_Hd(
				eprot.logits,
				eprot_i.logits
			)

			new_proposal = int(Hd_move > 0)

			# Compute observables of proposed state
			obs, eprot = self._compute_observables(eprot, pars, backward=False)

			obs, eprot = self._compute_observables(eprot, pars, backward=False)

			# ============================================================
			# TEST GRADIENT FLOW
			# ============================================================

			obs_test, eprot_test = self._compute_observables(
				eprot,
				pars,
				backward=True,
			)

			print("U =", obs_test)
			print("requires_grad =", eprot_test.logits.requires_grad)
			print("grad is None =", eprot_test.logits.grad is None)
			print(
				"grad finite =",
				torch.isfinite(eprot_test.logits.grad).all().item()
			)
			print(
				"grad norm =",
				eprot_test.logits.grad.norm().item()
			)

			# ============================================================
			# TEST REFERENCE STRUCTURE TOKENS
			# ============================================================

			print(
				"ref structure tokens shape:",
				self.ref_structure_tokens.shape
			)
			print(
				"ref structure tokens dtype:",
				self.ref_structure_tokens.dtype
			)
			print(
				"min token:",
				self.ref_structure_tokens.min().item()
			)
			print(
				"max token:",
				self.ref_structure_tokens.max().item()
			)

			# ============================================================
			# TEST REFERENCE CE
			# ============================================================

			ref_probs = self.ref_eprot.get_probs(pars["T_sftm"])
			

			with torch.no_grad():
				ref_logits = self._get_structure_logits(ref_probs)

			ref_ce = compute_U_structure_ce(
				ref_logits[:, 1:-1, :],
				self.ref_structure_tokens,
			)

			print("Reference CE =", ref_ce.item())

			# ============================================================
			# FINE TEST
			# ============================================================

			eprot.logits.grad = None

			dU_acc = obs['U']-data['U']

			# Accept/reject the proposed move (metropolis) only if a new sequence was generated
			if new_proposal:
				p = torch.rand(1, device=self.generator.device, generator=self.generator.get()).item()

				if p <= math.exp(-(dU_acc+dK)/pars['T']):

					# Proposed state accepted
					data = merge_dict(
						obs,
						data,
						overwrite=True,
					)

					data['accepted'] = 1
					data['muts_move'] = Hd_move

					eprot_i = eprot.copy()
					accepted_total += 1
					data["acc_moves"] = accepted_total

				else:

					# Proposed state rejected: keep observables of the previously accepted state
					data['accepted'] = 0
					data['muts_move'] = Hd_move

					eprot = eprot_i.copy()
			else:
				#no new sequence was generated: keep observables of the previously accepted state
				data['accepted'] = None
				data['muts_move'] = 0
				eprot = eprot_i.copy()

			# Complete update of data dictionary
			data['move'] = move
			data['new_proposal'] = new_proposal
			data['acc_moves'] = accepted_total
			data['acc_rate'] = accepted_total / move

			data['time'] = ptime() - self.t0

			data['dK'] = dK
			data['dU_int'] = dU_int
			data['dU_acc'] = dU_acc

			data['dE_int'] = dU_int + dK
			data['dE_acc'] = dU_acc + dK

			self._extend_buffer(data)

			# Save and/or print
			#test
			#if move%settings['log_step'] == 0:
			#	self._save_log(eprot, data, settings)
			if move%settings['print_step'] == 0:
				self._print_status(data)
        
		del self.ref_eprot, self.log, self.generator, self.t0



	def _setup(
			self,
			ref_seq: str,
			pars: dict,
			settings: dict | None = None,
	):
		# 1. PARS
		# First the inputted pars dictionary is checked, verifying the all the necessary keys are present.
		# Then, the pars dictionary is completed, adding missing keys and checking the type for the inputted ones.
		# Finally, the range of the values are checked for each parameters instance.
		assert is_subset(pars.keys(), self.defpars.keys()), f"{self.name}._setup(): unexpected key in inputted pars dictionary. Expected keys are: {list(self.defpars.keys())}."
		for key, (value, typ) in self.defpars.items():
			if value is None:
				assert key in pars.keys(), f"{self.name}._setup(): necessary key '{key}' missing from the inputted pars dictionary."
			else:
				if key not in pars:
					pars[key] = value
				else:
					try:
						pars[key] = typ(pars[key])
					except ValueError:
						raise ValueError(f"{self.name}.setup(): pars '{key}' type should be {typ}, but found {type(pars[key])}.")

		assert all([v>=0. for k,v in pars.items() if k in ["lambda_structure_ce", "lambda_S", "init_muts"]]), (
			f'{self.name}._setup(): invalid value for one of the following keys ("lambda_structure_ce", "lambda_S", "init_muts"). Allowed values: v>=0.'
		)
		assert (pars["lambda_structure_ce"]>0.) or (pars["lambda_S"]>0.), (
			f'{self.name}._setup(): invalid value for the keys "lambda_structure_ce" ({pars["lambda_structure_ce"]}) and "lambda_S" ({pars["lambda_S"]}). At least one must be positive.'
		)
		assert all([v>0. for k,v in pars.items() if k in ["moves", "T", "dt", "isteps", "M", "T_sftm", "eps"]]), (
			f'{self.name}._setup(): invalid value for one of the following keys ("moves", "T", "dt", "isteps", "M", "T_sftm", "eps"). Allowed values: v>0.'
		)
		pars = self._correct_types(pars, "pars")

		# 2. SETTINGS
		# First inputted settings are controlled and completed. Then, data, log and print steps are rounded to be divisible.
		# Finally threads and devices are set. Once the device is defined, the model, the generator and the reference extended protein are loaded on the correct device.
		if settings is not None:
			assert is_subset(settings.keys(), self.defsettings.keys()), f"{self.name}._setup(): unexpected key in inputted settings dictionary. Expected keys are: {list(self.defsettings.keys())}."
		else:
			settings = {}
		for key, (value, typ) in self.defsettings.items():
			if key not in settings:
				settings[key] = value
				if key in ["results_dir", "weights_dir"]:
					create_path(value)
			else:
				try:
					settings[key] = typ(settings[key])
				except ValueError:
					raise ValueError(f"{self.name}.setup(): settings '{key}' type should be {typ}, but found {type(settings[key])}.")

		assert settings["log_step"] > 0 and settings["print_step"] > 0, (
			f"{self.name}._setup(): 'steps' keys in settings must all be positive, "
			f"but found {settings['log_step']} ('log') and {settings['print_step']} ('print')."
		)

		assert settings["num_threads"] <= 5, f"{self.name}.setup(): invalid value for 'num_threads' variable {settings['num_threads']}. Allowed values: num_threads <= 5."
		torch.set_num_threads(settings["num_threads"])

		settings["device"] = torch.device(settings["device"]) if isinstance(settings["device"], str) else settings["device"]
		if ("cuda" in settings["device"].type) and (not torch.cuda.is_available()):
			settings["device"] = torch.device("cpu")

		self.model.to(settings["device"])
		self.generator = CustomGenerator(
				seed=pars["seed"],
				device=settings["device"],
		)

		self.ref_eprot = ExtendedProtein(
			sequence=ref_seq,
			requires_grad=False,
			device=self.generator.device
		)

		self.ref_eprot.expand()

		#self.ref_eprot.am = self.model.predict_attention(sequence_probs=self.ref_eprot.get_probs())

		# ---------------------------------------------------------
		# Reference structure tokens
		# ---------------------------------------------------------

		ref_probs = self.ref_eprot.get_probs(pars["T_sftm"])
		if ref_probs.ndim == 2:
			ref_probs = ref_probs.unsqueeze(0)

		with torch.no_grad():		#token is just a constant, no need to compute gradients
			ref_structure_logits = self._get_structure_logits(ref_probs)

			self.ref_structure_tokens = ref_structure_logits.argmax(	#struttura riferimento: argmax
				dim=-1
			)[:, 1:-1]	#escludiamo BOS e EOS tokens


		# 3. CLEAN
		# If restart is True (and everything required to restart a previous simulation exists), load the log information.
		# Otherwise, reset everything and proceed to clean every file (except for pars.txt) in results and weights directories.
		# The last two temporary attributes are here initiated, (log, and t0).
		if settings["restart"]:
			check_rdir = all([f in os.listdir(settings['results_dir']) for f in ['data.dat', 'pars.txt', 'generator.npy', 'log.pt']])
			check_edir = len([f for f in os.listdir(settings['eprot_dir']) if os.path.isfile(f"{settings['eprot_dir']}/{f}")]) >= 4
			settings["restart"] = check_rdir and check_edir

		if settings["restart"]:
			self.log = torch.load(f'{settings["results_dir"]}/log.pt')

			data = self.log["data"].copy()
			data = self._correct_types(data, "data")

			self.generator.load(f"{settings['results_dir']}/{self.log['generator']}")
			eprot = ExtendedProtein(
					device=self.generator.device,
					requires_grad=True
			)
			eprot.load(dir=settings['eprot_dir'], suffix=f"_{data['move']}")

			print(eprot.logits)
			print(eprot.tokens)
			print(eprot.sequence)
			#print(eprot.am)

		else:
			for d in [settings['results_dir'], settings['eprot_dir']]:
				for fn in os.listdir(d):
					if fn == 'pars.txt': continue
					if os.path.isfile(f'{d}/{fn}'):
						os.remove(f'{d}/{fn}')

			sequence = mutate(self.ref_eprot.sequence, pars['init_muts'], self.generator.get()) if pars["init_muts"]>0 else self.ref_eprot.sequence
			eprot = ExtendedProtein(sequence=sequence, requires_grad=True, device=self.generator.device)
			eprot.expand()

			print(eprot.logits)
			print(eprot.tokens)
			print(eprot.sequence)
			#print(eprot.am)
			
			data = {
				'move': 0,
				'time': 0.,
				'U': 0.,
				'U_structure_ce': 0.,
				'entropy': 0.,
				'Hd_to_ref': 0,
				'muts_move': 0,
				'new_proposal': 0,
				'dK': 0.,
				'dU_int': 0.,
				'dU_acc': 0.,
				'dE_int': 0.,
				'dE_acc': 0.,
				'accepted': 0,
				'acc_moves': 0,
				'acc_rate': 1.,
			}

			obs, eprot = self._compute_observables(eprot, pars, backward=False)
			data = merge_dict(
					obs,
					data,
			)
			data = self._correct_types(data, "data")
			self._extend_buffer(data, header=True)
			#test#self._save_log(eprot, data, settings)


		self.t0 = ptime() - data["time"]
		self._print_pars(pars, settings)
		self._print_status(data, header=True)

		return (
			eprot,
			data,
			pars,
			settings,
		)


	def _get_structure_logits(self, probs):

		if probs.ndim == 2:
			probs = probs.unsqueeze(0)

		# Match the dtype expected by the ESM3 sequence embedding
		probs = probs.to(dtype=self.model.encoder.sequence_embed.weight.dtype)

		#TEST
		print("probs dtype:", probs.dtype)
		print(
			"embedding dtype:",
			self.model.encoder.sequence_embed.weight.dtype
		)

		L = probs.shape[1]

		default_tokens, affine, affine_mask = self.model._default(
			L + 2,
			probs.device,
		)

		default_tokens = list(default_tokens)
		default_tokens[1] = default_tokens[1].to(torch.bfloat16)
		default_tokens[2] = default_tokens[2].to(torch.bfloat16)
		default_tokens = tuple(default_tokens)

		affine = affine.to(dtype=torch.bfloat16)

		x = self.model.encoder.custom_forward(
			probs,
			*default_tokens,
		)

		with torch.autocast(
			device_type="cuda",
			dtype=torch.bfloat16,
			enabled=probs.device.type == "cuda",
		):
			x, embedding, _ = self.model.transformer(
				x,
				sequence_id=None,
				affine=affine,
				affine_mask=affine_mask,
				chain_id=None,
			)

			structure_logits = self.model.output_heads.structure_head(x)

		return structure_logits


	def _extract_and_integrate(
			self,
			eprot: ExtendedProtein,
			pars: dict,
	):
		momenta = self._extract_momenta(eprot.logits.shape, pars['T'], pars['M'])
		K_i = self._compute_K(momenta, pars['M'])

		U_i, eprot = self._compute_observables(eprot, pars, backward=True)
		old_grad = eprot.logits.grad.detach().clone()
		eprot.logits.grad = None
    
		for istep in range(pars['isteps']):
			with torch.no_grad():
				eprot.logits += momenta*pars['dt']/pars['M'] - old_grad*pars['dt']**2./(2.*pars['M'])
				correction = eprot.logits.mean(axis=-1, keepdim=True)
				eprot.logits -= correction
			eprot.logits.requires_grad = True

			U_f, eprot = self._compute_observables(eprot, pars, backward=True)
			new_grad = eprot.logits.grad.detach().clone()
			eprot.logits.grad = None

			with torch.no_grad():
				momenta -= (new_grad+old_grad)*pars['dt']/2. + correction*pars['M']/pars['dt']
				Correction = momenta.mean(axis=-1, keepdim=True)
				momenta -= Correction

			if istep+1 < pars['isteps']:
				old_grad = new_grad.detach().clone()

		K_f = self._compute_K(momenta, pars['M'])
		return eprot, U_f-U_i, K_f-K_i



	def _extract_momenta(self, shape, T, M):
		p = torch.randn(*shape, device=self.generator.device, generator=self.generator.get()) * math.sqrt(T*M)
		p -= p.mean(axis=-1, keepdim=True)
		return p

	def _compute_observables(self, eprot, pars, backward=False):
		# integration
		if backward:

			probs = eprot.get_probs(pars['T_sftm'])

			structure_logits = self._get_structure_logits(probs)

			U_structure_ce = compute_U_structure_ce(
				structure_logits[:, 1:-1, :],
				self.ref_structure_tokens,
			)

			entropy = compute_entropy(
				probs,
				pars['eps']
			)

			U = (													#energia totale
				pars['lambda_structure_ce'] * U_structure_ce
				+ pars["lambda_S"] * entropy
			)

			U.backward(retain_graph=True)							#backpropagation: calcolo gradiente di U rispetto a logits (logits.grad)

			return U.item(), eprot
        
		# sampling										#accettazione
		else:

			probs = eprot.get_probs(pars['T_sftm'])

			structure_logits = self._get_structure_logits(probs)

			U_structure_ce = compute_U_structure_ce(
				structure_logits[:, 1:-1, :],
				self.ref_structure_tokens,
			)

			U = (
				pars['lambda_structure_ce'] * U_structure_ce
			)

			entropy = compute_entropy(
				probs,
				pars['eps']
			)

			Hd_to_ref = compute_Hd(
				eprot.logits,
				self.ref_eprot.logits
			)

			return {
				'U': U.item(),
				'U_structure_ce': U_structure_ce.item(),
				'entropy': entropy.item(),
				'Hd_to_ref': Hd_to_ref
			}, eprot
			
    ########
	def _compute_K(self, p, m):
		return (p**2.).sum().item() / (2.*m)



	def _extend_buffer(self, dikt, header=False):
		if header:
			# Explanation of data.dat columns
			self.buffer.write(
				"# data.dat column definitions:\n"
				"# move        : Monte Carlo move number\n"
				"# time        : CPU time elapsed since simulation start [s]\n"
				"# U           : potential energy of the current accepted sequence\n"
				"# U_structure_ce        : cross entropy contribution to U\n"
				"# entropy     : Shannon entropy of the softmax probabilities\n"
				"# Hd_to_ref   : Hamming distance between current accepted sequence and reference sequence\n"
				"# muts_move   : Hamming distance between proposed sequence and previous accepted sequence\n"
				"# new_proposal: 1 if integration changes the discrete sequence, 0 otherwise\n"
				"# dK          : change in kinetic energy during integration, K_f - K_i\n"
				"# dU_int      : change in potential energy during integration, U_f - U_i\n"
				"# dU_acc      : potential-energy difference between proposed and accepted sequence\n"
				"# dE_int      : total energy change during integration, dU_int + dK\n"
				"# dE_acc      : energy difference used in Metropolis acceptance, dU_acc + dK\n"
				"# accepted    : 1 if proposed move is accepted, 0 otherwise\n"
				"# acc_moves   : cumulative number of accepted Monte Carlo moves\n"
				"# acc_rate    : cumulative Monte Carlo acceptance rate, acc_moves / move\n"
				"#\n"
			)


			header, line = '', ''
			for key in dikt:
				header = header + f'{key}\t'
				line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{header[:-1]}\n{line[:-1]}\n")
		else:
			line = ''
			for key in dikt: line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{line[:-1]}\n")

	def _flush_buffer(self, settings):
		with open(f'{settings["results_dir"]}/data.dat', 'a') as f:
			print(self.buffer.getvalue(), file=f, end="")
		self.buffer.seek(0)
		self.buffer.truncate(0)

	def _save_log(self, eprot, data, settings):
		self._flush_buffer(settings)
		eprot.save(dir=settings["eprot_dir"], suffix=f'_{data["move"]}')
		self.generator.save(f'{settings["results_dir"]}/generator.npy')

		self.log = {
			"data": data.copy(),
			"generator": 'generator.npy',
		}
		torch.save(self.log, f'{settings["results_dir"]}/log.pt')

	def _print_status(self, data, header=False):
		if header:
			print(f'// {self.name} status register:')
			print(f'{self.separator}\n{self.header}\n{self.separator}')

		data['time_h'] = data["time"] / 3600.

		line = ''
		for key, _, fp in self.formatter['sampling']:
			line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'

		line = f'{line}|' + ''.join([' ']*5)

		for key, _, fp in self.formatter['efficiency']:
			value = data[key]

			if value is None:
				value = "None"
			else:
				value = format(value, f".{fp}f")

			line = f'{line}|{value:^12}'

		data.pop('time_h')

		print(f'{line}\n{self.separator}')


	def _print_pars(self, pars, settings):
		lines = []
		lines.append(f'# {self.name} parameters summary:')
		lines.append(f'# ')
		lines.append(f'# moves:                      {pars["moves"]:.1e}')
		lines.append(f'# sampling temperature:       {pars["T"]:.1e}')
		lines.append(f'# integration time step:      {pars["dt"]:.1e}')
		lines.append(f'# per-move integration steps: {pars["isteps"]:.0f}')
		lines.append(f'# logits mass:                {pars["M"]:.2f}')
		lines.append(f'# softmax temperature:        {pars["T_sftm"]:.2f}')
		lines.append(f'# attention multiplier:       {pars["lambda_structure_ce"]:.1e}')
		lines.append(f'# entropy multiplier:         {pars["lambda_S"]:.1e}')
		lines.append(f'# ')
		lines.append(f'# results directory: {settings["results_dir"]}')
		lines.append(f'# eprot directory:   {settings["eprot_dir"]}')
		lines.append(f'# restart:           {bool(settings["restart"])}')
		lines.append(f'# ')

		max_length = max([len(line) for line in lines])
		print('\n')
		print(''.join(['#'] * (max_length+2)))
		for line in lines:
			line = line + ''.join([' '] * (max_length-len(line)+1)) + '#'
			print(line)
		print(''.join(['#'] * (max_length+2)))
		print()



	def _correct_types(self, d, dname):
		# pars dictionary
		if dname == "pars":
			types_and_keys = [(int, ['moves', 'isteps', 'init_muts', 'seed'])]
		# data dictionary
		else:
			types_and_keys = [
				(int, [
					'move',
					'acc_moves',
					'Hd_to_ref',
					'muts_move',
					'new_proposal'
				])
			]

		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])
		return d

	def _init_attributes(self):
		self.buffer = StringIO()

		self.defpars = {
				"moves":(None, int),
				"T": (None, float),
				"dt": (1.0, float),
				"isteps": (100, int),
				"M": (1.0, float),
				"T_sftm": (0.1, float),
				"lambda_structure_ce": (1.0, float),
				"lambda_S": (0.0, float),
				"init_muts": (0, int),
				"eps": (1.0e-9, float),
				"seed": (0, int),
		}

		self.defsettings = {
				"results_dir": ("./results", str),
				"eprot_dir": ("./results/eprot", str),
				"log_step": (1, int),
				"print_step": (1, int),
				"restart": (False, bool),
				"device": ("cpu", str),
				"num_threads": (1, int),
		}

		self.formatter = {
			'sampling': [
				['move',             'move',             0],
				['U',                'U',                5],
				['U_structure_ce',   'U_structure_ce',   5],
				['entropy',          'entropy',          5],
				['Hd_to_ref',		 'Hd_to_ref',        0],
				['new_proposal', 	 'new_prop',	     0],
				['muts_move',        'muts_move',        0],
			],
			'efficiency': [
				['dE_int',           'dE_int',           5],
				['dE_acc',           'dE_acc',           5],
				['accepted',         'accepted',         0],
				['acc_rate',  		 'acc_rate',	     3],
			],
		}

		self.separator = ''.join(['-']*(13*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
