# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : optim_factory.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import json

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
	parameter_group_names = {}
	parameter_group_vars = {}

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights
		if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
			group_name = "no_decay"
			this_weight_decay = 0.
		else:
			group_name = "decay"
			this_weight_decay = weight_decay
		if get_num_layer is not None:
			layer_id = get_num_layer(name)
			group_name = "layer_%d_%s" % (layer_id, group_name)
		else:
			layer_id = None

		if group_name not in parameter_group_names:
			if get_layer_scale is not None:
				scale = get_layer_scale(layer_id)
			else:
				scale = 1.

			parameter_group_names[group_name] = {
				"weight_decay": this_weight_decay,
				"params": [],
				"lr_scale": scale
			}
			parameter_group_vars[group_name] = {
				"weight_decay": this_weight_decay,
				"params": [],
				"lr_scale": scale
			}
		parameter_group_vars[group_name]["params"].append(param)
		parameter_group_names[group_name]["params"].append(name)
	print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
	return list(parameter_group_vars.values())

def get_optimizer(net, cfg):
	weight_decay = cfg.TRAIN.WEIGHT_DECAY

	skip = {}
	if hasattr(net, 'no_weight_decay'):
		skip = net.no_weight_decay()
	param_dicts = get_parameter_groups(net, weight_decay, skip)
	weight_decay = 0.

	opt_args = dict(lr=cfg.TRAIN.LR, weight_decay=weight_decay)

	if cfg.TRAIN.OPT_EPS is not None:
		opt_args['eps'] = cfg.TRAIN.OPT_EPS
	# if cfg.TRAIN.OPT_BETAS is not None:
	# 	opt_args['betas'] = cfg.TRAIN.OPT_BETAS
	print("optimizer settings:", opt_args)

	opt_lower = cfg.TRAIN.OPTIMIZER.lower()
	if opt_lower == 'sgd' or opt_lower == 'nesterov':
		opt_args.pop('eps', None)
		optimizer = torch.optim.SGD(param_dicts, momentum=cfg.TRAIN.MOMENTUN, nesterov=True, **opt_args)
	elif opt_lower == 'momentum':
		opt_args.pop('eps', None)
		optimizer = torch.optim.SGD(param_dicts, momentum=cfg.TRAIN.MOMENTUN, nesterov=False, **opt_args)
	elif opt_lower == 'adam':
		optimizer = torch.optim.Adam(param_dicts, **opt_args)
	elif opt_lower == 'adamw':
		optimizer = torch.optim.AdamW(param_dicts, **opt_args)
	else:
		raise ValueError("Unsupported Optimizer")

	return optimizer
