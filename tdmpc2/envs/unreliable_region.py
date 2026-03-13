import gymnasium as gym
import numpy as np
import torch


class UnreliableRegionWrapper(gym.Wrapper):
	"""Randomize actions when end-effector enters a designated bad region.

	Wraps a raw ManiSkill env (before ManiSkillWrapper in the wrapper chain).
	Accesses tcp pose directly from ManiSkill's internal robot state.

	The bad region is defined by a spatial axis and threshold:
	  - axis: 0=x, 1=y, 2=z of the EE position
	  - threshold: boundary value on that axis
	  - side: "positive" means EE > threshold is bad,
	          "negative" means EE < threshold is bad
	"""

	def __init__(self, env, axis=1, threshold=0.0, side="positive"):
		super().__init__(env)
		self.axis = axis
		self.threshold = threshold
		self.side = side

	def _in_bad_region(self):
		"""Check if EE is currently in the bad region."""
		ee_pos = self.unwrapped.agent.tcp.pose.p[0]  # (3,) tensor, drop batch dim
		val = ee_pos[self.axis].item()
		if self.side == "positive":
			return val > self.threshold
		return val < self.threshold

	def step(self, action):
		if self._in_bad_region():
			if isinstance(action, torch.Tensor):
				action = torch.rand_like(action) * 2 - 1
			elif isinstance(action, np.ndarray):
				action = np.random.uniform(-1, 1, size=action.shape).astype(action.dtype)
		return self.env.step(action)
