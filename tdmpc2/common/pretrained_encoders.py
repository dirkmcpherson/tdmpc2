import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from common.layers import ShiftAug, NormedLinear, SimNorm


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

BACKBONE_SPECS = {
	'dino':     {'out_dim': 384, 'input_size': 224},
	'r3m':      {'out_dim': 512, 'input_size': 64},
	'resnet18': {'out_dim': 512, 'input_size': 64},
}


def _load_backbone(encoder_type):
	"""Load a pretrained backbone and return (model, out_dim)."""
	if encoder_type == 'dino':
		model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
		return model, 384
	elif encoder_type == 'r3m':
		from r3m import load_r3m
		r3m_model = load_r3m('resnet18')
		r3m_model.eval()
		return r3m_model, 512
	elif encoder_type == 'resnet18':
		from torchvision import models
		resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
		# Remove the final FC layer, use avgpool output
		modules = list(resnet.children())[:-1]  # everything except fc
		model = nn.Sequential(*modules, nn.Flatten())
		return model, 512
	else:
		raise ValueError(f"Unknown encoder type: {encoder_type}")


class ChannelAdapter(nn.Module):
	"""
	Splits multi-channel input (4ch, 6ch, etc.) into 3ch groups,
	zero-pads the last group if needed, passes each through the backbone
	independently, and concatenates outputs.
	Applies ImageNet normalization per 3ch group.
	"""

	def __init__(self, backbone, in_channels):
		super().__init__()
		self.backbone = backbone
		self.in_channels = in_channels
		self.num_groups = (in_channels + 2) // 3  # ceil division

		# ImageNet normalization buffers
		self.register_buffer('img_mean', IMAGENET_MEAN.view(1, 3, 1, 1))
		self.register_buffer('img_std', IMAGENET_STD.view(1, 3, 1, 1))

	def _normalize(self, x):
		"""Apply ImageNet normalization to a 3-channel tensor in [0, 1]."""
		return (x - self.img_mean) / self.img_std

	def forward(self, x):
		# x: (B, C, H, W) in [0, 255]
		x = x.float() / 255.0

		groups = []
		for i in range(self.num_groups):
			start = i * 3
			end = min(start + 3, self.in_channels)
			group = x[:, start:end]
			# Zero-pad if fewer than 3 channels
			if group.shape[1] < 3:
				pad = torch.zeros(group.shape[0], 3 - group.shape[1], group.shape[2], group.shape[3],
								  device=group.device, dtype=group.dtype)
				group = torch.cat([group, pad], dim=1)
			group = self._normalize(group)
			groups.append(self.backbone(group))

		return torch.cat(groups, dim=-1)


class PretrainedEncoder(nn.Module):
	"""
	Drop-in replacement for conv() encoder using pretrained backbones.
	Supports DINOv2, R3M, and ResNet-18.
	"""

	def __init__(self, cfg, obs_shape):
		super().__init__()
		encoder_type = cfg.encoder
		spec = BACKBONE_SPECS[encoder_type]
		in_channels = obs_shape[0]
		input_size = spec['input_size']
		backbone_out_dim = spec['out_dim']

		# Number of 3ch groups determines total output dim
		num_groups = (in_channels + 2) // 3
		total_dim = backbone_out_dim * num_groups

		# Load backbone
		raw_backbone, _ = _load_backbone(encoder_type)

		# Build preprocessing: ShiftAug at 64x64, then optional resize
		preprocess = [ShiftAug()]
		if input_size != 64:
			preprocess.append(
				nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)
			)
		self.preprocess = nn.Sequential(*preprocess)

		# Channel adapter wraps backbone to handle multi-channel input
		self.channel_adapter = ChannelAdapter(raw_backbone, in_channels)

		# Projection head: NormedLinear -> Linear -> SimNorm
		self.projection = nn.Sequential(
			NormedLinear(total_dim, cfg.latent_dim),
			nn.Linear(cfg.latent_dim, cfg.latent_dim),
			SimNorm(cfg),
		)

		# Freeze backbone if requested
		if cfg.freeze_encoder:
			for p in self.channel_adapter.backbone.parameters():
				p.requires_grad = False

		# Save a copy of backbone weights to restore after weight_init
		self._backbone_weights = deepcopy(self.channel_adapter.backbone.state_dict())

	def save_backbone_weights(self):
		"""Save current backbone weights (call before weight_init if needed)."""
		self._backbone_weights = deepcopy(self.channel_adapter.backbone.state_dict())

	def restore_backbone_weights(self):
		"""Restore pretrained backbone weights (call after weight_init)."""
		self.channel_adapter.backbone.load_state_dict(self._backbone_weights)

	def forward(self, x):
		x = self.preprocess(x)
		x = self.channel_adapter(x)
		return self.projection(x)
