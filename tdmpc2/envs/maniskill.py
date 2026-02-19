import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from envs.wrappers.timeout import Timeout

import mani_skill.envs


MANISKILL_TASKS = {
	'pick-cube': dict(
		env='PickCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'stack-cube': dict(
		env='StackCube-v1',
		control_mode='pd_ee_delta_pos',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v1',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v1',
		control_mode='pd_ee_delta_pose',
	),
}

# Conv encoder requires 64x64 images
IMG_SIZE = 64


class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if cfg.obs == 'state':
			obs_shape = self.env.observation_space.shape[1:]  # strip batch dim
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
			)
		else:  # rgb: 3ch base_camera RGB + 1ch hand_camera depth, channels-first
			self.observation_space = gym.spaces.Box(
				low=0, high=255, shape=(4, IMG_SIZE, IMG_SIZE), dtype=np.uint8
			)
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=np.float32,
		)

	def _extract_image(self, obs):
		"""Extract (4, 64, 64) uint8 image: base_camera RGB + hand_camera depth."""
		sensor_data = obs['sensor_data']
		cameras = list(sensor_data.values())

		# Concatenate RGB from all cameras along channel dim → (B, H, W, 3*N)
		# Take base_camera (first 3 channels), drop batch dim → (H, W, 3)
		rgb = torch.cat([c['rgb'] for c in cameras], dim=-1)[0, :, :, :3].float()

		# Concatenate depth from all cameras → (B, H, W, N)
		# Take hand_camera (index 1), drop batch dim → (H, W, 1)
		depth = torch.cat([c['depth'] for c in cameras], dim=-1)[0, :, :, 1:2].float()
		depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
		depth = torch.clamp(depth / 2000.0 * 255.0, 0, 255)

		# Resize both to IMG_SIZE: (H, W, C) → (1, C, H, W) → interpolate → (C, H, W)
		rgb_t = F.interpolate(
			rgb.permute(2, 0, 1).unsqueeze(0),
			size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
		).squeeze(0)
		depth_t = F.interpolate(
			depth.permute(2, 0, 1).unsqueeze(0),
			size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
		).squeeze(0)

		# Combine to (4, H, W) uint8
		return torch.cat([rgb_t, depth_t], dim=0).byte().cpu().numpy()

	def _extract_obs(self, obs):
		if self.cfg.obs == 'state':
			return obs.squeeze(0).cpu().numpy()
		return self._extract_image(obs)

	def reset(self):
		obs, _ = self.env.reset()
		return self._extract_obs(obs)

	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, terminated, truncated, info = self.env.step(action)
			reward += r.item()
			done = bool((terminated | truncated).item())
			if done:
				break
		obs = self._extract_obs(obs)
		info['terminated'] = bool(terminated.item())
		info['success'] = bool(info['success'].item())

		if info['success']:
			reward += 100.

		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self):
		return self.env.render()


def make_env(cfg):
	"""
	Make ManiSkill3 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs in ('state', 'rgb'), 'This task only supports state or rgb observations.'
	task_cfg = MANISKILL_TASKS[cfg.task]
	if cfg.obs == 'state':
		env = gym.make(
			task_cfg['env'],
			obs_mode='state',
			control_mode=task_cfg['control_mode'],
			num_envs=1,
			render_mode='rgb_array',
		)
	else:  # rgb
		env = gym.make(
			task_cfg['env'],
			obs_mode='rgb+depth',
			control_mode=task_cfg['control_mode'],
			num_envs=1,
			robot_uids='panda_wristcam',
			render_mode='rgb_array',
		)
	env = ManiSkillWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	return env
