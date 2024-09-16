import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
import os
import tempfile

class Buffer():
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda')
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon+1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity
    
    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f'Buffer capacity: {self._capacity:,}')
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum([
                (v.numel()*v.element_size() if not isinstance(v, TensorDict) \
                else sum([x.numel()*x.element_size() for x in v.values()])) \
            for v in tds.values()
        ]) / len(tds)
        total_bytes = bytes_per_step*self._capacity
        print(f'Storage required: {total_bytes/1e9:.2f} GB')
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = 'cuda' if 2.5*total_bytes < mem_free else 'cpu'
        print(f'Using {storage_device.upper()} memory for storage.')
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (arg.to(device, non_blocking=True) \
            if arg is not None else None for arg in args)

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = td['obs']
        action = td['action'][1:]
        reward = td['reward'][1:].unsqueeze(-1)
        task = td['task'][0] if 'task' in td.keys() else None
        return self._to_device(obs, action, reward, task)

    def add(self, td):
        """Add an episode to the buffer."""
        td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
        return self._prepare_batch(td)

    def save(self, path):
        """
        Save the buffer's contents to a file.
        
        :param path: The file path to save the buffer.
        """
        if self._buffer is None:
            print("Buffer is empty, nothing to save.")
            return

        S = self._capacity
        permanent_dir = path  # Specify your desired directory path here

        # Create the directory if it doesn't exist
        os.makedirs(permanent_dir, exist_ok=True)

        # Save the buffer to the permanent directory
        self._buffer.dumps(permanent_dir)
        print(f"Buffer saved to: {permanent_dir}")

        # sampler = RandomSampler()
        storage = LazyTensorStorage(S)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )

        rb_load = self._reserve_buffer(storage)
        rb_load.loads(permanent_dir)

        assert len(self._buffer) == len(rb_load)

    def load(self, path, capacity=None):
        """
        Load the buffer's contents from a file.
        
        :param path: The file path to load the buffer from.
        """
        if not os.path.exists(path):
            print(f"directory {path} does not exist.")
            return False
        if capacity is None: capacity = self._capacity
        return True

  # sampler = RandomSampler()
        storage = LazyTensorStorage(capacity)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )

        rb_load = self._reserve_buffer(storage)
        rb_load.loads(path)
        self._buffer = rb_load

        print(f"Buffer loaded from {path}")

    def compare_buffers(buffer1, buffer2):
        assert buffer1.capacity == buffer2.capacity, "Capacities don't match"
        assert buffer1.num_eps == buffer2.num_eps, "Number of episodes don't match"
        
        # Sample from both buffers and compare
        sample1 = buffer1.sample()
        sample2 = buffer2.sample()
        
        for key in sample1:
            if sample1[key] is not None and sample2[key] is not None:
                assert torch.allclose(sample1[key], sample2[key]), f"{key} doesn't match"
        
        print("Buffers match!")