import torch
import numpy as np


class CustomGenerator:
	def __init__(
		self,
		seed: int | None = 0,
		device: str | torch.device | None = "cpu",
	):
		self.device = device if isinstance(device, torch.device) else torch.device(device)
		self.seed = seed if isinstance(seed, int) else 0
		self.generator = torch.Generator(device)
		self.generator.manual_seed(self.seed)

	def get(self):
		return self.generator

	def save(self, fn: str):
		state = self.generator.get_state().detach().cpu().numpy()
		with open(fn, 'wb') as f:
			np.save(f, state)

	def load(self, fn: str):
		with open(fn, 'rb') as f:
			state = np.load(f, allow_pickle=True)
		state = torch.tensor(state, dtype=torch.uint8)
		try:
			self.generator.set_state(state)
		except RuntimeError:
			print(f"CustomGenerator.load(): RNG state is wrong size (i.e. state device in file {fn} is not compatible with generator device {self.device}). Load ignored.")
