#!/usr/bin/env python3
"""
Convert ManiSkill h5 demonstrations to tdmpc2 npz format.

Usage:
    # State observations (default)
    python convert_maniskill.py --h5 /path/to/demos.h5 --type pick-cube

    # RGB+depth observations (matches obs=rgb training)
    python convert_maniskill.py --h5 /path/to/demos.h5 --type pick-cube --obs rgb

Output files are written to tdmpc2/demonstrations/{type}/ and contain:
    obs:        (T+1, state_dim) float32          — state mode
                (T+1, 4, 64, 64) uint8            — rgb mode (base_camera RGB + hand_camera depth, channels-first)
    action:     (T+1, action_dim) float32          — first step is NaN
    reward:     (T+1,) float32                     — first step is NaN
    terminated: (T+1,) float32                     — first step is NaN
"""

import io
import argparse
import pathlib

import h5py
import numpy as np
from PIL import Image
import tqdm

IMG_SIZE = 64  # must match tdmpc2/envs/maniskill.py


def extract_state(traj_obs):
    """Extract flat state from trajectory observation group.

    Tries traj['obs']['state'] first (recorded with obs_mode='state').
    Falls back to concatenating agent + scalar extra fields.
    """
    if 'state' in traj_obs:
        return traj_obs['state'][:].astype(np.float32)  # (T+1, state_dim)

    parts = []
    agent = traj_obs['agent']
    for k in sorted(agent.keys()):
        v = agent[k][:]
        if v.ndim == 3:
            v = v[:, 0, :]  # squeeze batch dim
        parts.append(v.reshape(v.shape[0], -1).astype(np.float32))

    if 'extra' in traj_obs:
        extra = traj_obs['extra']
        for k in sorted(extra.keys()):
            v = extra[k][:]
            if v.ndim == 3:
                v = v[:, 0, :]
            v = v.reshape(v.shape[0], -1).astype(np.float32)
            if v.shape[-1] <= 64:
                parts.append(v)

    return np.concatenate(parts, axis=-1)  # (T+1, state_dim)


def _norm_depth(depth):
    """Normalize raw depth to [0, 255] uint8 (2 m range)."""
    max_mm = 2000.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=max_mm, neginf=0.0)
    return np.clip(depth / max_mm * 255.0, 0, 255).astype(np.uint8)


def _resize_rgb(frames):
    """Resize (T, H, W, 3) uint8 RGB to (T, IMG_SIZE, IMG_SIZE, 3)."""
    out = []
    for frame in frames:
        img = Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        out.append(np.array(img))
    return np.stack(out, axis=0)


def _resize_depth(frames):
    """Resize (T, H, W, 1) uint8 depth to (T, IMG_SIZE, IMG_SIZE, 1)."""
    out = []
    for frame in frames:
        d = Image.fromarray(frame[:, :, 0], mode='L').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        out.append(np.array(d)[:, :, None])
    return np.stack(out, axis=0)


def _cam_parts(second_cam):
    """Return list of (camera, modality) pairs — mirrors envs/maniskill.py logic."""
    if second_cam == 'none':
        return [('base', 'rgb'), ('hand', 'depth')]
    elif second_cam == 'rgb':
        return [('base', 'rgb'), ('hand', 'rgb')]
    elif second_cam == 'depth':
        return [('base', 'depth'), ('hand', 'depth')]
    elif second_cam == 'rgbd':
        return [('base', 'rgb'), ('base', 'depth'), ('hand', 'rgb'), ('hand', 'depth')]
    else:
        raise ValueError(f'Unknown second_cam={second_cam}')


def extract_image(traj_obs, second_cam='none'):
    """Extract image obs matching tdmpc2's _extract_image.

    Channel layout is determined by second_cam (see _cam_parts).
    Depth is normalized to [0, 255] uint8 (2 m range), matching the live env.
    """
    sd = traj_obs['sensor_data']
    cam_names = {'base': 'base_camera', 'hand': 'hand_camera'}
    parts = []
    for cam, modality in _cam_parts(second_cam):
        h5_cam = cam_names[cam]
        assert h5_cam in sd and modality in sd[h5_cam], (
            f"'{h5_cam}/{modality}' not found. Re-record demos with obs_mode='rgb+depth' "
            "and robot_uids='panda_wristcam'."
        )
        raw = sd[h5_cam][modality][:]
        if modality == 'rgb':
            parts.append(_resize_rgb(raw))
        else:
            parts.append(_resize_depth(_norm_depth(raw.astype(np.float32))))

    combined = np.concatenate(parts, axis=-1)  # (T+1, H, W, C)
    return combined.transpose(0, 3, 1, 2)      # (T+1, C, H, W) uint8


def convert_h5_to_npz(h5_path, output_dir, obs_mode='state', second_cam='none', add_success_reward=False):
    h5_path = pathlib.Path(h5_path).expanduser()
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    print_sample = True
    with h5py.File(h5_path, 'r') as f:
        traj_keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))

        for traj_key in tqdm.tqdm(traj_keys, desc=f"Converting {h5_path.name}"):
            traj = f[traj_key]

            if obs_mode == 'rgb':
                obs = extract_image(traj['obs'], second_cam=second_cam)   # (T+1, C, 64, 64) uint8
            else:
                obs = extract_state(traj['obs'])   # (T+1, state_dim) float32

            action = traj['actions'][:].astype(np.float32)     # (T, action_dim)
            reward = traj['rewards'][:].astype(np.float32).reshape(-1)  # (T,)
            terminated = traj['terminated'][:].astype(np.float32).reshape(-1)  # (T,)

            if add_success_reward and 'success' in traj:
                success = traj['success'][:].reshape(-1).astype(bool)
                reward[success] += 100.0

            # Prepend NaN for t=0 (initial obs has no prior action/reward/terminated)
            full_action = np.concatenate(
                [np.full((1, action.shape[1]), float('nan'), dtype=np.float32), action], axis=0
            )
            full_reward = np.concatenate([[float('nan')], reward]).astype(np.float32)
            full_terminated = np.concatenate([[float('nan')], terminated]).astype(np.float32)

            episode = {
                'obs':        obs,
                'action':     full_action,
                'reward':     full_reward,
                'terminated': full_terminated,
            }

            traj_id = traj_key.split('_')[1]
            filename = output_dir / f"traj_{traj_id}.npz"

            if print_sample:
                # print out the keys shapes and dtypes of the first trajectory for verification
                print(f"Sample trajectory keys and shapes from {filename}:")
                for k, v in episode.items():
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                print("...")  # indicate that this is just a sample
                print_sample = False

            with io.BytesIO() as buf:
                np.savez_compressed(buf, **episode)
                buf.seek(0)
                with open(filename, 'wb') as out:
                    out.write(buf.read())

    print(f"Saved {len(traj_keys)} trajectories to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=False, default=None, help='Path to ManiSkill h5 demo file')
    parser.add_argument('--type', required=True, help='Demo generation source', choices=['teleop', 'motionplanning'])
    parser.add_argument('--obs', default='state', choices=['state', 'rgb'],
                        help='Observation mode: state (default) or rgb (base_camera RGB + hand_camera depth)')
    parser.add_argument('--second_cam', default='none', choices=['none', 'rgb', 'depth', 'rgbd'],
                        help='Extra channels from second camera: none (default), rgb (hand RGB +3ch), '
                             'depth (base depth +1ch), rgbd (both +4ch)')
    parser.add_argument('--add_success_reward', action='store_true',
                        help='Add +100 reward bonus on steps where success=True')
    parser.add_argument('--task', default="PickCube-v1", help='Task name for default h5 path (ignored if --h5 is provided)')
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    suffix = args.obs if args.second_cam == 'none' else f'{args.obs}_{args.second_cam}'
    output_dir = script_dir / 'demonstrations' / args.type / suffix
    
    if args.h5 is None:
        print("No --h5 path provided, using default location based on --type:")
        h5 = pathlib.Path.home() / '.maniskill' / 'demos' / f'{args.task}' / f'{args.type}' / f'trajectory.state+rgb+depth.pd_ee_delta_pos.physx_cpu.h5'
    else:
        h5 = pathlib.Path(args.h5).expanduser()

    # assert h5 file exists before starting conversion
    if not h5.is_file():
        raise FileNotFoundError(f"H5 file not found: {h5}. Please check the --h5 path and task type.")

    convert_h5_to_npz(h5, output_dir, obs_mode=args.obs, second_cam=args.second_cam,
                       add_success_reward=args.add_success_reward)
