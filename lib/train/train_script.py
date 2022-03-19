import os
# loss function related
from torch.nn import BCEWithLogitsLoss, MSELoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .data.dataset_loader import *
from .optim_factory import *
from .schedule_factory import *
from lib.utils.misc import get_world_size
# network related
from lib.models.mae_3d_model import build_mae_3d
from lib.models.swin.swin_transformer import build_swin
from lib.models.vit.vit import build_vit
# forward propagation related
from lib.train.actors import MAE3DACTOR
# for import modules
import importlib


def run(settings):
	settings.description = 'Training script'

	# update the default configs with config file
	if not os.path.exists(settings.cfg_file):
		raise ValueError("%s doesn't exist." % settings.cfg_file)
	config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
	cfg = config_module.cfg
	config_module.update_config_from_file(settings.cfg_file)
	if settings.local_rank in [-1, 0]:
		print("New configuration is shown below.")
		for key in cfg.keys():
			print("%s configuration:" % key, cfg[key])
			print('\n')

	# update dataset settings based on cfg
	update_settings(settings, cfg)

	# Record the training log
	log_dir = os.path.join(settings.save_dir, 'logs')
	if settings.local_rank in [-1, 0]:
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
	settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

	# Build dataloaders
	loader_train, loader_val = build_seq_dataloaders(cfg, settings)
	# Create network
	net = build_vit(cfg.MODEL)

	# checkpoint_path = os.path.abspath(
	# 	os.path.join(os.getcwd(), "../..")) + '/checkpoints/pretrain_mae_vit_base_mask_0.75_400e.pth'
	checkpoint_path = os.path.abspath(
		os.path.join(os.getcwd(), "../..")) + '/checkpoints/MAE3D_ep0010_2loss.pth.tar'
	net.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['net'], strict=True)
	print("load pretrain: " + checkpoint_path)
	net.cuda()

	# wrap networks to distributed one
	if settings.local_rank != -1:
		net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
		settings.device = torch.device("cuda:%d" % settings.local_rank)
	else:
		settings.device = torch.device("cuda:0")

	# Loss functions and Actors
	if settings.script_name == "mae":
		objective = {'mse': MSELoss()}
		loss_weight = {'mse': cfg.TRAIN.MSE_WEIGHT}
		actor = MAE3DACTOR(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
	else:
		raise ValueError("illegal script name")

	# Optimizer, parameters, and learning rates
	optimizer = get_optimizer(net, cfg)
	lr_scheduler = get_schedule(cfg, optimizer)

	trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
	trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True)
