# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : mae_decoder.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import torch.nn as nn
from einops import rearrange
from .module import PatchEmbed, Block, trunc_normal_, get_norm_layer
from .position_encoding import get_sinusoid_encoding_table

class MAE3DEncoder(nn.Module):
	def __init__(self, img_size=224, patch_size=16, in_chans=3, seq_len = 10, num_classes=0, embed_dim=768, depth=12,
	             num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
	             drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_learnable_pos_emb=False):
		super().__init__()
		self.num_classes = num_classes
		self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
		self.seq_len = seq_len

		self.patch_embed = PatchEmbed(
			img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
		num_patches = self.patch_embed.num_patches * self.seq_len

		# TODO: Add the cls token
		# self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
		else:
			# sine-cosine positional embeddings
			self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
		self.blocks = nn.ModuleList([
			Block(
				dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
				drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
				init_values=init_values)
			for i in range(depth)])
		self.norm = norm_layer(embed_dim)
		self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

		if use_learnable_pos_emb:
			trunc_normal_(self.pos_embed, std=.02)

		# trunc_normal_(self.cls_token, std=.02)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def get_num_layers(self):
		return len(self.blocks)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed', 'cls_token'}

	def get_classifier(self):
		return self.head

	def reset_classifier(self, num_classes, global_pool=''):
		self.num_classes = num_classes
		self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

	def forward_features(self, x, mask):
		x = rearrange(x, 'b t c h w -> (b t) c h w')
		mask = rearrange(mask, 'b t h w -> (b t) h w')
		x, mask = self.patch_embed(x, mask)
		# cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		# x = torch.cat((cls_tokens, x), dim=1)
		x = rearrange(x, '(b t) p n -> b (t p) n', t=self.seq_len)
		mask = rearrange(mask, '(b t) p-> b (t p)', t = self.seq_len)

		x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

		B, _, C = x.shape
		# x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
		x_vis = x.reshape(B, -1, C)

		for blk in self.blocks:
			x_vis = blk(x_vis, mask)

		x_vis = self.norm(x_vis)
		return x_vis

	def forward(self, x, mask):
		x = self.forward_features(x, mask)
		x = self.head(x)
		return x


def build_encoder(cfg):
	return MAE3DEncoder(
		img_size=cfg.IMAGE_SIZE,
		patch_size=cfg.PATCH_SIZE,
		in_chans=cfg.ENCODER.IN_CHANNELS,
		num_classes=cfg.ENCODER.NUM_CLASSES,
		embed_dim=cfg.ENCODER.EMBED_DIM,
		depth=cfg.ENCODER.DEPTH,
		num_heads=cfg.ENCODER.NUM_HEADS,
		mlp_ratio=cfg.ENCODER.MLP_RATIO,
		qkv_bias=cfg.ENCODER.QKV_BIAS,
		seq_len=cfg.SEQ_LEN,
		# qk_scale=cfg.ENCODER.QK_SCALE,
		qk_scale=None,
		drop_rate=cfg.ENCODER.DROP_RATE,
		attn_drop_rate=cfg.ENCODER.ATTN_DROP_RATE,
		drop_path_rate=cfg.ENCODER.DROP_PATH_RATE,
		norm_layer=get_norm_layer(cfg.ENCODER.NORM_LAYER),
		init_values=cfg.ENCODER.INIT_VALUES,
		use_learnable_pos_emb=cfg.ENCODER.USE_LEARNABLE_POS_EMB)
