# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : module.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, drop_path
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from functools import partial
from lib.utils.image import *


def trunc_normal_(tensor, mean=0., std=1.):
	__call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def get_norm_layer(type):
	if type == "layer_norm":
		return partial(nn.LayerNorm, eps=1e-6)
	return None


class DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
	"""

	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training)

	def extra_repr(self) -> str:
		return 'p={}'.format(self.drop_prob)


class PatchEmbed(nn.Module):
	""" Image to Patch Embedding
	"""

	def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)
		num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
		self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
		self.img_size = img_size
		self.patch_size = patch_size
		self.num_patches = num_patches

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x, mask, **kwargs):
		B, C, H, W = x.shape
		# FIXME look at relaxing size constraints
		assert H == self.img_size[0] and W == self.img_size[1], \
			f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
		x = self.proj(x).flatten(2).transpose(1, 2)
		out_size = [H // self.patch_size[0], W // self.patch_size[1]]
		mask = F.interpolate(mask[None].float(), size=out_size, mode='bilinear').flatten(2).squeeze(0)
		return x, mask.to(torch.bool)


class Attention(nn.Module):
	def __init__(
			self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
			proj_drop=0., attn_head_dim=None):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		if attn_head_dim is not None:
			head_dim = attn_head_dim
		all_head_dim = head_dim * self.num_heads
		self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5

		self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
		if qkv_bias:
			self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
			self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
		else:
			self.q_bias = None
			self.v_bias = None

		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(all_head_dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, attn_mask=None):
		# x: (B, N, C)
		# attn_mask: (B,N)
		B, N, C = x.shape
		qkv_bias = None
		if self.q_bias is not None:
			qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
		# qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
		qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))
		if attn_mask is not None:
			attn = attn.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		# x = self.drop(x)
		# commit this for the orignal BERT implement
		x = self.fc2(x)
		x = self.drop(x)
		return x


class Block(nn.Module):
	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
	             drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
	             attn_head_dim=None):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
			attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

		if init_values > 0:
			self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
			self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
		else:
			self.gamma_1, self.gamma_2 = None, None

	def forward(self, x, mask=None):
		if self.gamma_1 is None:
			x = x + self.drop_path(self.attn(self.norm1(x), mask))
			x = x + self.drop_path(self.mlp(self.norm2(x)))
		else:
			x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask))
			x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
		return x
