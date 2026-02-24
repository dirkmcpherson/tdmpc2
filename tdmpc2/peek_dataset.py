"""
Peek at TD-MPC2 datasets (.npz episode files).

Usage:
    # Print value ranges for all arrays
    python peek_dataset.py demonstrations/teleop --ranges

    # Play back RGB frames with reward/action overlays
    python peek_dataset.py demonstrations/teleop

    # Show per-dimension action histograms
    python peek_dataset.py demonstrations/teleop --actions
"""

import argparse
import glob
import os
import time
import numpy as np


# --------------------------------------------------------------------------- #
#  Ranges mode
# --------------------------------------------------------------------------- #

def print_ranges(directory, max_files=None):
    files = _gather_files(directory)
    if not files:
        return

    count = 0
    for fp in files:
        if max_files and count >= max_files:
            break
        count += 1
        print(f"\nFile: {os.path.basename(fp)}")
        try:
            with np.load(fp) as data:
                _print_dict_ranges(dict(data))
        except Exception as e:
            print(f"  Error: {e}")


def _print_dict_ranges(d):
    for key in sorted(d.keys()):
        val = np.asarray(d[key])
        if np.issubdtype(val.dtype, np.number) or np.issubdtype(val.dtype, np.bool_):
            print(f"  {key:55s} | shape: {str(val.shape):20s} | dtype: {str(val.dtype):8s} | min: {val.min():10.4f} | max: {val.max():10.4f} | mean: {val.mean():10.4f}")
        else:
            print(f"  {key:55s} | shape: {str(val.shape):20s} | dtype: {val.dtype}")


# --------------------------------------------------------------------------- #
#  Video playback mode
# --------------------------------------------------------------------------- #

def _depth_to_colormap(depth_img):
    """Convert a single-channel depth image to a BGR colormap for display.
    depth_img: (H, W) uint8 array (already normalized to 0-255).
    """
    import cv2
    return cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO)


def _obs_to_display_frame(obs, cell_size=256):
    """Convert a channels-first obs array to a displayable BGR frame.

    Handles various channel counts:
      3ch  → single RGB
      4ch  → base RGB (3) + hand depth (1), side by side
      6ch  → base RGB (3) + hand RGB (3), side by side
      other → split into 3ch groups, show side by side
    """
    import cv2

    # obs is (C, H, W) uint8, channels-first
    C, H, W = obs.shape
    panels = []

    if C == 3:
        # Single RGB
        img = np.transpose(obs, (1, 2, 0))  # (H, W, 3)
        panels.append(('RGB', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    elif C == 4:
        # 3ch RGB + 1ch depth
        rgb = np.transpose(obs[:3], (1, 2, 0))
        panels.append(('Base RGB', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
        depth = obs[3]  # (H, W) uint8
        panels.append(('Hand Depth', _depth_to_colormap(depth)))
    elif C == 6:
        # Two 3ch RGB images
        rgb1 = np.transpose(obs[:3], (1, 2, 0))
        panels.append(('Base RGB', cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)))
        rgb2 = np.transpose(obs[3:6], (1, 2, 0))
        panels.append(('Hand RGB', cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)))
    else:
        # Generic: show 3ch groups as RGB, remaining as grayscale
        idx = 0
        panel_num = 0
        while idx < C:
            remaining = C - idx
            if remaining >= 3:
                img = np.transpose(obs[idx:idx+3], (1, 2, 0))
                panels.append((f'Ch {idx}-{idx+2}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
                idx += 3
            else:
                for j in range(remaining):
                    ch = obs[idx + j]
                    panels.append((f'Ch {idx+j}', cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)))
                idx += remaining
            panel_num += 1

    # Resize and arrange side by side
    resized = []
    for label, panel in panels:
        panel = cv2.resize(panel, (cell_size, cell_size))
        cv2.putText(panel, label, (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        resized.append(panel)

    return np.concatenate(resized, axis=1)


def play_files(directory, pause_time=2.0, fps=30, max_files=None):
    import cv2

    files = _gather_files(directory)
    if not files:
        return

    print(f"Found {len(files)} files. Press 'q' to quit, space to pause/resume.")
    count = 0

    for fp in files:
        if max_files and count >= max_files:
            break
        count += 1
        print(f"Playing: {os.path.basename(fp)}")

        try:
            with np.load(fp) as npz:
                if 'obs' not in npz:
                    print(f"  Skipping: no 'obs' key. Keys: {list(npz.keys())}")
                    continue

                obs_data = npz['obs']
                # obs could be state (2D) or image (4D channels-first)
                if obs_data.ndim != 4:
                    print(f"  Skipping: obs is not image data (shape={obs_data.shape})")
                    continue

                rewards = npz.get('reward', None)
                actions = npz.get('action', None)
                terminated = npz.get('terminated', None)

                num_frames = len(obs_data)
                for i in range(num_frames):
                    frame = _obs_to_display_frame(obs_data[i])
                    h = frame.shape[0]

                    # Overlays
                    y = h - 10
                    if terminated is not None and i < len(terminated) and terminated[i]:
                        cv2.putText(frame, "TERMINATED", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                        y -= 25
                    if actions is not None and i < len(actions):
                        a_str = "[" + ", ".join(f"{x:.2f}" for x in actions[i]) + "]"
                        cv2.putText(frame, f"Act: {a_str}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                        y -= 25
                    if rewards is not None and i < len(rewards):
                        cv2.putText(frame, f"Reward: {float(rewards[i]):.2f}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame, f"Frame {i}/{num_frames}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                    cv2.imshow("TD-MPC2 Dataset Peek", frame)
                    key = cv2.waitKey(int(1000 / fps)) & 0xFF
                    if key == ord("q"):
                        print("Quitting...")
                        cv2.destroyAllWindows()
                        return
                    elif key == ord(" "):
                        # Pause until space pressed again
                        while True:
                            key2 = cv2.waitKey(0) & 0xFF
                            if key2 == ord(" ") or key2 == ord("q"):
                                break
                        if key2 == ord("q"):
                            cv2.destroyAllWindows()
                            return

            print(f"  Finished {os.path.basename(fp)}. Pausing {pause_time}s...")
            time.sleep(pause_time)

        except Exception as e:
            print(f"  Error reading {fp}: {e}")

    cv2.destroyAllWindows()
    print("Done processing all files.")


# --------------------------------------------------------------------------- #
#  Action histogram mode
# --------------------------------------------------------------------------- #

def plot_action_histograms(directory, max_files=None):
    import matplotlib.pyplot as plt

    files = _gather_files(directory)
    if not files:
        return

    print(f"Aggregating actions from {len(files)} files...")
    all_actions = []
    count = 0

    for fp in files:
        if max_files and count >= max_files:
            break
        count += 1
        try:
            with np.load(fp) as npz:
                if 'action' in npz:
                    all_actions.append(npz['action'])
        except Exception as e:
            print(f"  Error reading {fp}: {e}")

    if not all_actions:
        print("No actions found.")
        return

    all_actions = np.concatenate(all_actions, axis=0)
    if all_actions.ndim == 1:
        all_actions = all_actions[:, np.newaxis]
    if all_actions.ndim != 2:
        print(f"Unexpected action shape: {all_actions.shape}")
        return

    num_dims = all_actions.shape[1]
    print(f"Plotting histograms for {num_dims} action dims. Total samples: {len(all_actions)}")

    cols = min(4, num_dims)
    rows = (num_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    axes = axes.flatten()

    for i in range(num_dims):
        ax = axes[i]
        dim_actions = all_actions[:, i]
        ax.hist(dim_actions, bins=50, color="skyblue", edgecolor="black")
        ax.set_title(f"Action Dim {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        stats = f"min: {dim_actions.min():.2f}\nmax: {dim_actions.max():.2f}\nmean: {dim_actions.mean():.2f}"
        ax.text(0.95, 0.95, stats, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    for i in range(num_dims, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  File discovery
# --------------------------------------------------------------------------- #

def _gather_files(directory):
    """Find .npz files in a directory (non-recursive)."""
    npzs = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not npzs:
        print(f"No .npz files found in {directory}")
    else:
        print(f"Found {len(npzs)} npz files in {directory}")
    return npzs


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Peek at TD-MPC2 datasets (.npz episode files)."
    )
    parser.add_argument("path", type=str, help="Directory containing .npz files")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds between files (default: 2.0)")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--ranges", action="store_true", help="Print value ranges instead of playing video")
    parser.add_argument("--actions", action="store_true", help="Show action histograms")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to process")

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: '{args.path}' is not a directory.")
    elif args.actions:
        plot_action_histograms(args.path, args.max_files)
    elif args.ranges:
        print_ranges(args.path, args.max_files)
    else:
        play_files(args.path, args.pause, args.fps, args.max_files)
