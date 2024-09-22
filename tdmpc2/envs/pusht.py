import gymnasium as gym
import gym_pusht as pusht
import numpy as np

from envs.wrappers.time_limit import TimeLimit

class PushT(gym.Env):
    def __init__(self, size=(64,64), obs_type="pixels_state", render_mode="rgb_array", force_sparse=False, max_steps=1000, action_repeat=2):
        w,h = size
        self._env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode, observation_width=w, observation_height=h, force_sparse=force_sparse, display_cross=False)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = "image"
        self.force_sparse = force_sparse # Force the reward to be sparse
        self.max_steps = max_steps; self.nstep = 0
        self.max_episode_steps = max_steps
        self.action_repeat = action_repeat
        
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)
        
    def step(self, action):
        action = np.clip((action + 1.0) / 2.0 * 512, 0, 512)

        for _ in range(self.action_repeat):
            obs, reward, done, truncated, info = self._env.step(action)
            if done: break

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        else:
            for k,v in obs.items():
                obs[k] = np.array(v)

        if "image" not in obs and "pixels" in obs:
            obs["image"] = obs["pixels"]

        info['success'] = np.array(info.get('is_success', False))
        info['coverage'] = np.array(info.get('coverage', 0.0))

        if info["success"]:
            reward = 2 * self.max_steps # self.max_steps - self.nstep
            print("Success!")
        # elif info['coverage'] > 0.5:
        #     print(f"Coverage: {info['coverage']:1.2f}")

        if self.force_sparse:
            reward = 1.0 if info['success'] else 0.0

        # Transpose 
        obs["is_first"] = np.array(False)
        obs["is_last"] = np.array(done)
        obs["is_terminal"] = np.array(info.get("is_terminal", False))

        self.nstep += 1
        return obs, np.array(reward), np.array(done), info

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}

        # replace pixels with image
        if "pixels" in spaces:
            spaces["image"] = spaces.pop("pixels")
            
        return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )
    
    @property
    def action_space(self):
        return gym.spaces.Box(np.array([-0.5, -0.5], dtype=np.float32), np.array([0.5, 0.5], dtype=np.float32), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        if "image" not in obs and "pixels" in obs:
            obs["image"] = obs["pixels"]

        obs["is_first"] = np.array(True)
        obs["is_last"] = np.array(False)
        obs["is_terminal"] = np.array(False)
        self.nstep = 0
        return obs

    def render(self, mode="human", width=64, height=64):
        return self._env.render()

    def close(self):
        self._env.close()

def make_env(cfg):
    if 'pusht' not in cfg.task.lower():
        raise ValueError(f"Task {cfg.task} is not supported by the PushT environment.")
    env = PushT(max_steps=300, force_sparse=cfg.force_sparse)
    env = TimeLimit(env, max_episode_steps=300)
    return env

if __name__ == '__main__':
    env = PushT(max_steps=300)