import torch

from custom_esm.sdk.api import ESM3InferenceClient, ESMProteinTensor, GenerationConfig

from .matrices import compute_contact_map


def init_structure_config(
		track: str = "structure",
		invalid_ids: list[int] = [],
		schedule: str = "cosine",
		strategy: str = "entropy",
		temperature_annealing: bool = False,
		num_steps: int = 1,
		temperature: float = 0.0,
		top_p: float = 1.0,
		condition_on_coordinates_only: bool = True,
		**kwargs,
) -> GenerationConfig: 
	return GenerationConfig(
			track=track,
			invalid_ids=invalid_ids,
			schedule=schedule,
			strategy=strategy,
			temperature_annealing=temperature_annealing,
			num_steps=num_steps,
			temperature=temperature,
			top_p=top_p,
			condition_on_coordinates_only=condition_on_coordinates_only,
	)


def predict_structure(
		model: ESM3InferenceClient,
		tokens: ESMProteinTensor,
		config: GenerationConfig,
		d0: float = 8.0,
		infer_cbeta: bool = True,
		to: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
	tensor = ESMProteinTensor(sequence=tokens)
	prot = model.predict_protein(tensor, config, infer_cbeta)
	cm = compute_contact_map(prot.coordinates, d0)
	if to is not None:
		cm = cm.to(to)
		prot.plddt = prot.plddt.to(to)
	return cm, prot.plddt
