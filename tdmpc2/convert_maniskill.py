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


def extract_image(traj_obs):
    """Extract (T+1, 4, 64, 64) uint8 image matching tdmpc2's _extract_image.

    Layout: base_camera RGB (3ch) + hand_camera depth (1ch), channels-first.
    Depth is normalized to [0, 255] uint8 (2 m range), matching the live env.
    """
    sd = traj_obs['sensor_data']

    assert 'base_camera' in sd and 'rgb' in sd['base_camera'], (
        "'base_camera/rgb' not found. Re-record demos with obs_mode including rgb "
        "and robot_uids='panda_wristcam'."
    )
    assert 'hand_camera' in sd and 'depth' in sd['hand_camera'], (
        "'hand_camera/depth' not found. Re-record demos with robot_uids='panda_wristcam'."
    )

    base_rgb = sd['base_camera']['rgb'][:]    # (T+1, H, W, 3) uint8
    hand_depth = sd['hand_camera']['depth'][:].astype(np.float32)  # (T+1, H, W, 1)

    # Normalize depth to [0, 255] uint8 (2 m range), matching _extract_image
    max_mm = 2000.0
    hand_depth = np.nan_to_num(hand_depth, nan=0.0, posinf=max_mm, neginf=0.0)
    hand_depth = np.clip(hand_depth / max_mm * 255.0, 0, 255).astype(np.uint8)

    # Resize each frame with PIL (LANCZOS), matching dreamerv3-torch conversion
    resized_rgb = []
    for frame in base_rgb:
        img = Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        resized_rgb.append(np.array(img))
    resized_rgb = np.stack(resized_rgb, axis=0)  # (T+1, H, W, 3)

    resized_depth = []
    for frame in hand_depth:
        d = Image.fromarray(frame[:, :, 0], mode='L').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        resized_depth.append(np.array(d)[:, :, None])
    resized_depth = np.stack(resized_depth, axis=0)  # (T+1, H, W, 1)

    # Combine and convert to channels-first (T+1, 4, H, W)
    combined = np.concatenate([resized_rgb, resized_depth], axis=-1)  # (T+1, H, W, 4)
    return combined.transpose(0, 3, 1, 2)  # (T+1, 4, H, W) uint8


def convert_h5_to_npz(h5_path, output_dir, obs_mode='state', add_success_reward=False):
    h5_path = pathlib.Path(h5_path).expanduser()
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        traj_keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))

        for traj_key in tqdm.tqdm(traj_keys, desc=f"Converting {h5_path.name}"):
            traj = f[traj_key]

            if obs_mode == 'rgb':
                obs = extract_image(traj['obs'])   # (T+1, 4, 64, 64) uint8
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

            with io.BytesIO() as buf:
                np.savez_compressed(buf, **episode)
                buf.seek(0)
                with open(filename, 'wb') as out:
                    out.write(buf.read())

    print(f"Saved {len(traj_keys)} trajectories to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True, help='Path to ManiSkill h5 demo file')
    parser.add_argument('--type', required=True, help='Task type name, e.g. pick-cube')
    parser.add_argument('--obs', default='state', choices=['state', 'rgb'],
                        help='Observation mode: state (default) or rgb (base_camera RGB + hand_camera depth)')
    parser.add_argument('--add_success_reward', action='store_true',
                        help='Add +100 reward bonus on steps where success=True')
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    output_dir = script_dir / 'demonstrations' / args.type / args.obs

    convert_h5_to_npz(args.h5, output_dir, obs_mode=args.obs, add_success_reward=args.add_success_reward)
