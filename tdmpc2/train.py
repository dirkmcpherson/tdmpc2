import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage

torch.backends.cudnn.benchmark = True
import cv2

@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	env = make_env(cfg)

	buf = Buffer(cfg)
	load_buffer = Buffer(cfg) # HACK
	if cfg.demo_path:
		buf._buffer = buf._reserve_buffer(LazyTensorStorage(buf.capacity, device=torch.device('cpu')))
		buf._num_eps += 1
		
		load_buffer._capacity = 10000
		load_buffer._buffer = load_buffer._reserve_buffer(LazyTensorStorage(10000, device=torch.device('cpu')))
		if not (load_buffer.load(os.path.expanduser(cfg.demo_path))):
			raise FileNotFoundError(f"Could not load buffer at {cfg.demo_path}")

		print(f"Loading from {cfg.demo_path}, transitions (3 frames): {len(load_buffer._buffer.storage)}")

		from tensordict.tensordict import TensorDict
		def to_td(obs, action=None, reward=None):
			"""Creates a TensorDict for a new episode."""
			if isinstance(obs, dict):
				obs = TensorDict(obs, batch_size=(), device='cpu')
			else:
				obs = obs.unsqueeze(0).cpu()
			if action is None:
				action = torch.full_like(env.rand_act(), float('nan'))
			if reward is None:
				reward = torch.tensor(float('nan'))
			td = TensorDict(dict(
				obs=obs,
				action=action.unsqueeze(0),
				reward=reward.unsqueeze(0),
			), batch_size=(1,))
			return td

		# show the demos:
		for i in range(len(load_buffer._buffer.storage)):
			obs, action, reward = load_buffer._buffer[i]["obs"], load_buffer._buffer[i]["action"], load_buffer._buffer[i]["reward"]
			if 'cluster' not in cfg.demo_path: # dont show images on the cluster
				cv2.imshow(str(cfg.demo_path.split('/')[-1]), obs.detach().cpu().numpy().transpose(1,2,0)[:, :, 6:])
				cv2.waitKey(1)
			buf.add(to_td(obs, action, reward))

		# for _ in range(10):
		# 	obs, action, reward, task = load_buffer.sample()

		# 	td = TensorDict(dict(
		# 		obs=obs,
		# 		action=action.unsqueeze(0),
		# 		reward=reward.unsqueeze(0),
		# 	))

		# 	buf.add(td)

		# 	from IPython import embed as ipshell; ipshell()
		# 	cv2.imshow('buff example', obs.cpu().numpy()[:,:,6:9])
		# 	cv2.waitKey(10)


	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=buf,
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
