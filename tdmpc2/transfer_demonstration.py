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
import numpy as np
torch.backends.cudnn.benchmark = True
import cv2
import io

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

    buf = Buffer(cfg)
    load_buffer = Buffer(cfg) # HACK
    from pathlib import Path
    base_path = Path('~/workspace/tdmpc2/demonstrations/').expanduser()
    demo_dirs = ['HD_0_sparse', 'AD_1_sparse', 'HD_1_sparse', 'HD_2_sparse', 'HD_3', 'AD_2_sparse', 'AD_3_sparse']
    # demo_dirs = ['AD_1_sparse', 'AD_2_sparse', 'AD_3_sparse']
    for ddir in demo_dirs:
        demo_path = base_path / ddir
        device = 'cpu'
        
        load_buffer._capacity = cfg.demo_buffer_size
        load_buffer._buffer = load_buffer._reserve_buffer(LazyTensorStorage(cfg.demo_buffer_size, device=torch.device(device)))
        if not (load_buffer.load(os.path.expanduser(demo_path))):
            raise FileNotFoundError(f"Could not load buffer at {demo_path}")

        print(f"Loading from {demo_path}, transitions (3 frames): {len(load_buffer._buffer.storage)}")
        
        # show the demos:
        tds = []
        
        ep_keys = ["pixels", "image", "state", "is_first", "is_last", "is_terminal", "reward", "discount", "action"]
        ep_dict = {k: [] for k in ep_keys}

        r = torch.tensor(0.0, device=device)
        n_ep = load_buffer._buffer[0]['episode']; n_steps = 0;
        debug_max_len = -1; debug_read_eps = 0
        for i in range(len(load_buffer._buffer.storage)):
            obs, action, reward, episode = load_buffer._buffer[i]["obs"], load_buffer._buffer[i]["action"], load_buffer._buffer[i]["reward"], load_buffer._buffer[i]["episode"]
            # if 'cluster' not in str(demo_path): # dont show images on the cluster
            #     toshow = obs.detach().cpu().numpy().transpose(1,2,0); title = str(demo_path).split('/')[-1]
            #     cv2.imshow(title+'0', toshow[:, :, :3]); cv2.imshow(title+'1', toshow[:, :, 3:6]); cv2.imshow(title+'2', toshow[:, :, 6:])
            #     cv2.waitKey(1)

            # print(load_buffer._buffer[i].keys())
            # from IPython import embed as ipshell; ipshell()
            if episode != n_ep:
                if len(ep_dict['reward']) == 0:
                    from IPython import embed as ipshell; ipshell()
                else:
                    debug_max_len = max(debug_max_len, n_steps)

                    # write out:
                    ep_dict['is_last'][-1] = True if n_steps < 300 else False  # is_last is true if episode ends with success
                    ep_dict['is_terminal'][-1] = True if n_steps == 300 else False # terminal is last step of episode's time limit

                    out_path = Path(f"~/workspace/fastrl/logs/demonstrations/TDMPC_pusht_{ddir}/transfered/{n_ep}.npz").expanduser()
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    # save as an npz
                    # Convert all tensors to numpy arrays
                    np_ep_dict = {k: np.array([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in v]) for k, v in ep_dict.items()}

                    with io.BytesIO() as f1:
                        np.savez_compressed(f1, **np_ep_dict); f1.seek(0)
                        with out_path.open("wb") as f2:
                            f2.write(f1.read())
                    print(f"{n_ep} {n_steps=} last reward {ep_dict['reward'][-1]}. Total reward {r}. Next ep {episode}. Written to {out_path}")

                    r = torch.tensor(0.)
                    n_steps = 0
                debug_read_eps += 1
                n_ep = episode
                if cfg.debug and debug_read_eps > 5: 
                    print(f"Read {debug_read_eps} episodes"); break
            else:
                # put into fastrl expected format
                if all(torch.isnan(action)):
                    pass # skip the nan transition in the start
                else:
                    ep_dict['action'].append(action)
                    ep_dict['pixels'].append(obs[6:9, ...]) # just take the last of the 3 frames
                    ep_dict['image'].append(obs[6:9, ...]) # just take the last of the 3 frames
                    ep_dict['reward'].append(reward)
                    ep_dict['is_first'].append(False if i > 0 else True)
                    ep_dict['is_last'].append(False)
                    ep_dict['is_terminal'].append(False)
                    n_steps += 1
            if not torch.isnan(reward): r += reward


        assert debug_max_len < 301, f"Expected 300, got {debug_max_len}. Is this not pusht?"



if __name__ == '__main__':
    train()
