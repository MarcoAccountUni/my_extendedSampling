import torch
import numpy as np

from custom_esm.utils.constants import esm3 as C



""" ####################### """
""" OPERATIONS ON SEQUENCES """
""" ####################### """

# Calculate Hamming distance between two sequences (or logits tensors)
def compute_Hd(input_i: str | torch.Tensor, input_j: str | torch.Tensor):
	if isinstance(input_i, str) and isinstance(input_j, str):
		Hd = sum([amino_i != amino_j for amino_i, amino_j in zip(input_i, input_j)])
	elif isinstance(input_i, torch.Tensor) and isinstance(input_j, torch.Tensor):
		tokens_i = input_i.argmax(axis=-1)
		tokens_j = input_j.argmax(axis=-1)
		Hd = (tokens_i != tokens_j).sum().item()
	else:
		raise ValueError(f"compute_Hd(): inputs must both either str or torch.Tensor, found {type(input_i)} and {type(input_j)}.")
	return Hd

# Mutate ref_sequence for a given number of mutations
def mutate(ref_sequence: str, mutations: int, generator: torch.Generator):
	sites = torch.randint(low=0, high=len(ref_sequence), size=(mutations,), device=generator.device, generator=generator).tolist()
	idxs = torch.randint(low=0, high=len(C.SEQUENCE_USED_VOCAB), size=(mutations,), device=generator.device, generator=generator).tolist()
	aminos = [C.SEQUENCE_USED_VOCAB[idx] for idx in idxs]
    
	sequence = np.array(list(ref_sequence))
	sequence[sites] = aminos
	mutations = (sequence != np.array(list(ref_sequence))).sum()
	return "".join(sequence.tolist())



""" ########################## """
""" OPERATIONS ON DICTIONARIES """
""" ########################## """

# Merge two dictionaries
def merge_dict(from_dict, into_dict, overwrite=True):
	if overwrite:
		for key in from_dict:
			into_dict[key] = from_dict[key]
	else:
		for key in from_dict:
			if key not in into_dict.keys():
				into_dict[key] = from_dict[key]
	return into_dict

# Merge many dictionaries
def merge_dicts(from_dicts, into_dict, overwrite=True):
	for from_dict in from_dicts:
		into_dict = merge_dict(from_dict, into_dict, overwrite)
	return into_dict

# Check that the keys of dictionary A are a subset of the keys of dictionary B
def is_subset(keys_A, keys_B):
    if len(keys_A) > len(keys_B):
        return False
    else:
        return all([k in keys_B for k in keys_A])



""" #################### """
""" OPERATIONS ON FLOATS """
""" #################### """

# Get order of magnitude of input number
def get_ofm(x):
	return abs(np.log10(abs(x)).astype(int))+1

# Evaluate if two numbers are close
def isclose(a, b, rel_tol=1e-08, abs_tol=0.0):
	return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
