from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.cswintt_2.config import cfg, update_config_from_file
def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/cswintt_2/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # params.checkpoint = os.path.join(save_dir, "checkpoints/STARKST_ep0500.pth.tar")
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/cstrt_1/baseline_cs_all_dataset/CSTRT_ep0050.pth.tar")
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/cstrt_1/baseline_cs_expandtrain/CSTRT_ep0040.pth.tar")
    # params.checkpoint_cls = os.path.join(save_dir, "checkpoints/train/cstrt_1/baseline/CSTRT_ep0020.pth.tar")
    params.checkpoint_cls = os.path.join(save_dir, "checkpoints/train/cstrt_2/baseline_cs/CSTRT_final.pth")

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
