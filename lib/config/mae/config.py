from easydict import EasyDict as edict
import yaml

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.IMAGE_SIZE = 224
cfg.MODEL.PATCH_SIZE = 16
cfg.MODEL.SEQ_LEN = 5

# ENCODER
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.IN_CHANNELS = 3
cfg.MODEL.ENCODER.EMBED_DIM = 768
cfg.MODEL.ENCODER.DEPTH = 12
cfg.MODEL.ENCODER.NUM_HEADS = 12
cfg.MODEL.ENCODER.NUM_CLASSES = 0
cfg.MODEL.ENCODER.MLP_RATIO = 4
cfg.MODEL.ENCODER.QKV_BIAS = True
cfg.MODEL.ENCODER.DROP_RATE = 0.
cfg.MODEL.ENCODER.ATTN_DROP_RATE = 0.
cfg.MODEL.ENCODER.DROP_PATH_RATE = 0.
cfg.MODEL.ENCODER.NORM_LAYER = 'layer_norm'
cfg.MODEL.ENCODER.INIT_VALUES = 0.
cfg.MODEL.ENCODER.USE_LEARNABLE_POS_EMB = False

# DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.EMBED_DIM = 384
cfg.MODEL.DECODER.DEPTH = 4
cfg.MODEL.DECODER.NUM_HEADS = 6
cfg.MODEL.DECODER.NUM_CLASSES = 768
cfg.MODEL.DECODER.MLP_RATIO = 4
cfg.MODEL.DECODER.QKV_BIAS = True
cfg.MODEL.DECODER.DROP_RATE = 0.
cfg.MODEL.DECODER.ATTN_DROP_RATE = 0.
cfg.MODEL.DECODER.DROP_PATH_RATE = 0.
cfg.MODEL.DECODER.NORM_LAYER = 'layer_norm'
cfg.MODEL.DECODER.INIT_VALUES = 0.

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.OPT_EPS = 1e-8
cfg.TRAIN.MOMENTUN = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.05

cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WARMUP_LR = 0.000001

cfg.TRAIN.MSE_WEIGHT = 1.0

cfg.TRAIN.EPOCH = 500
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_WORKER = 1

# cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
# cfg.TRAIN.L1_WEIGHT = 5.0
# cfg.TRAIN.FREEZE_BACKBONE_BN = True
# cfg.TRAIN.FREEZE_LAYERS = ['conv1', 'layer1']
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "warmup_cos"
cfg.TRAIN.SCHEDULER.LR_STEP_SIZE = 100
cfg.TRAIN.SCHEDULER.LR_STEP_GAMMA = 0.75
cfg.TRAIN.SCHEDULER.WARMUP_EPOCH = 100
cfg.TRAIN.SCHEDULER.WARMUP_FACTOR = 0.2
cfg.TRAIN.SCHEDULER.WARMUP_FIANL_VALUE_FACTOR = 0.1
# cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
# cfg.DATA.SAMPLER_MODE = "trident_pro"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
# cfg.DATA.MAX_SAMPLE_INTERVAL = [200]
cfg.DATA.SIZE = 224
cfg.DATA.CENTER_JITTER = 2
cfg.DATA.SCALE_JITTER = 1

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_train_full" ,"TRACKINGNET"]
cfg.DATA.TRAIN.DATASETS_RATIO = None
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 50000

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = None
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 1000


#
# # DATA.SEARCH
# cfg.DATA.SEARCH = edict()
# cfg.DATA.SEARCH.NUMBER = 1  # number of search frames for multiple frames training
# cfg.DATA.SEARCH.SIZE = 384
# cfg.DATA.SEARCH.FACTOR = 5.0
# cfg.DATA.SEARCH.CENTER_JITTER = 4.5
# cfg.DATA.SEARCH.SCALE_JITTER = 0.5
#
# # DATA.TEMPLATE
# cfg.DATA.TEMPLATE = edict()
# cfg.DATA.TEMPLATE.NUMBER = 2
# cfg.DATA.TEMPLATE.SIZE = 128
# cfg.DATA.TEMPLATE.FACTOR = 2.0
# cfg.DATA.TEMPLATE.CENTER_JITTER = 0
# cfg.DATA.TEMPLATE.SCALE_JITTER = 0
#
# # TEST
# cfg.TEST = edict()
# cfg.TEST.TEMPLATE_FACTOR = 2.0
# cfg.TEST.TEMPLATE_SIZE = 128
# cfg.TEST.SEARCH_FACTOR = 5.0
# cfg.TEST.SEARCH_SIZE = 384
# cfg.TEST.EPOCH = 500
# cfg.TEST.UPDATE_INTERVALS = edict()
# cfg.TEST.UPDATE_INTERVALS.LASOT = [200]
# cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = [200]
# cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = [200]
# cfg.TEST.UPDATE_INTERVALS.VOT20 = [200]
# cfg.TEST.UPDATE_INTERVALS.VOT20LT = [200]


def _edict2dict(dest_dict, src_edict):
	if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
		for k, v in src_edict.items():
			if not isinstance(v, edict):
				dest_dict[k] = v
			else:
				dest_dict[k] = {}
				_edict2dict(dest_dict[k], v)
	else:
		return


def gen_config(config_file):
	cfg_dict = {}
	_edict2dict(cfg_dict, cfg)
	with open(config_file, 'w') as f:
		yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
	if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
		for k, v in exp_cfg.items():
			if k in base_cfg:
				if not isinstance(v, dict):
					base_cfg[k] = v
				else:
					_update_config(base_cfg[k], v)
			else:
				raise ValueError("{} not exist in config.py".format(k))
	else:
		return


def update_config_from_file(filename):
	exp_config = None
	with open(filename) as f:
		exp_config = edict(yaml.safe_load(f))
		_update_config(cfg, exp_config)
