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


def _cam_parts(cfg):
	"""Return list of (camera, modality) pairs for the observation channels.

	second_cam controls what the hand (second) camera contributes:
	  none  → base RGB (3) + hand depth (1) = 4ch  [backward compatible]
	  rgb   → base RGB (3) + hand RGB (3) = 6ch
	  depth → base depth (1) + hand depth (1) = 2ch
	  rgbd  → base RGBD (4) + hand RGBD (4) = 8ch
	"""
	second = getattr(cfg, 'second_cam', 'none')
	if second == 'none':
		return [('base', 'rgb'), ('hand', 'depth')]
	elif second == 'rgb':
		return [('base', 'rgb'), ('hand', 'rgb')]
	elif second == 'depth':
		return [('base', 'depth'), ('hand', 'depth')]
	elif second == 'rgbd':
		return [('base', 'rgb'), ('base', 'depth'), ('hand', 'rgb'), ('hand', 'depth')]
	else:
		raise ValueError(f'Unknown second_cam={second}')


def _img_channels(cfg):
	"""Number of observation image channels based on second_cam setting."""
	ch = {'rgb': 3, 'depth': 1}
	return sum(ch[mod] for _, mod in _cam_parts(cfg))


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
		else:  # rgb: base_camera RGB + hand_camera depth + optional second_cam channels
			n_ch = _img_channels(cfg)
			self.observation_space = gym.spaces.Box(
				low=0, high=255, shape=(n_ch, IMG_SIZE, IMG_SIZE), dtype=np.uint8
			)
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=np.float32,
		)

	def _resize(self, t):
		"""Resize (H, W, C) tensor to (C, IMG_SIZE, IMG_SIZE)."""
		return F.interpolate(
			t.permute(2, 0, 1).unsqueeze(0),
			size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
		).squeeze(0)

	def _norm_depth(self, depth):
		"""Normalize raw depth (mm) to [0, 255] float."""
		depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
		return torch.clamp(depth / 2000.0 * 255.0, 0, 255)

	def _extract_image(self, obs):
		"""Extract image obs based on second_cam config."""
		sd = obs['sensor_data']
		cams = {'base': sd['base_camera'], 'hand': sd['hand_camera']}
		parts = []
		for cam_name, modality in _cam_parts(self.cfg):
			raw = cams[cam_name][modality][0].float()  # (H, W, C), drop batch dim
			if modality == 'depth':
				raw = self._norm_depth(raw)
			parts.append(self._resize(raw))
		return torch.cat(parts, dim=0).byte().cpu().numpy()

	def _extract_obs(self, obs):
		if self.cfg.obs == 'state':
			return obs.squeeze(0).cpu().numpy()
		return self._extract_image(obs)

	def reset(self):
		obs, _ = self.env.reset()
		return self._extract_obs(obs)

	def _custom_success(self):
		"""Less strict success: grasped + within goal_thresh, no static robot requirement."""
		u = self.env.unwrapped
		obj_to_goal = u.goal_site.pose.p - u.cube.pose.p  # (B, 3)
		distance = float(torch.linalg.norm(obj_to_goal, dim=-1)[0].item())
		is_grasped = bool(u.agent.is_grasping(u.cube)[0].item())
		return is_grasped and distance < u.goal_thresh

	def step(self, action):
		reward = 0
		for _ in range(2):
			obs, r, terminated, truncated, info = self.env.step(action)
			reward += r.item()
			done = bool((terminated | truncated).item())
			if done:
				break
		info['terminated'] = bool(terminated.item())
		info['success'] = bool(info['success'].item())

		# Override with less strict success check (no static robot requirement)
		if not info['success'] and self._custom_success():
			info['success'] = True
			done = True

		if info['success']:
			reward += 100.

		obs = self._extract_obs(obs)
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
