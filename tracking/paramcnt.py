# from lib.models.layers.head import build_box_head
# import torch
# import importlib
# from lib.models.ostrack import build_ostrack, build_small_ostrack
# from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
# from lib.models.layers.attn_blocks import CEBlock
#
# yaml_fname = 'experiments/ostrack/ostrack_distillation_123_128_h64.yaml'
# config_module = importlib.import_module('lib.config.ostrack.config')
# cfg = config_module.cfg
# config_module.update_config_from_file(yaml_fname)
#
# model = build_small_ostrack(cfg)
# box_head = build_box_head(cfg, 128)
# backbone = vit_base_patch16_224_ce('', drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
#                                    ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
#                                    ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
#                                    depth=3,
#                                    channel=cfg.MODEL.BACKBONE.CHANNELS,
#                                    heads=cfg.MODEL.BACKBONE.HEADS
#                                    )
# # model = torch.nn.Conv2d(16, 8, (3, 3), stride=1, padding=1)
# # model = torch.nn.BatchNorm2d(16)
# block = CEBlock(dim=128, num_heads=4)
# n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(model)
# print(n_parameters)

from lib.models.efficientvit import build_efficienttrack
import argparse
import importlib
import torch

parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
parser.add_argument('--script', type=str, default='efficienttrack', help='Name of the train script.')
parser.add_argument('--config', type=str, default='experiments/efficienttrack/para_base_4_BN.yaml',
                    help="Name of the config file.")
args = parser.parse_args()

config_module = importlib.import_module("lib.config.%s.config" % args.script)
cfg = config_module.cfg
config_module.update_config_from_file(args.config)
model = build_efficienttrack(cfg)
ckpt = torch.load('/home/lpy/OSTrack/output/checkpoints/train/efficienttrack/para_base_4_BN/EfficientTrack_ep0300.pth.tar')['net']
a, b = model.load_state_dict(ckpt, strict=False)
def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)
replace_batchnorm(model)
template = torch.rand((1, 3, 128, 128))
search1 = torch.rand((1, 3, 256, 256))
search2 = torch.rand((1, 3, 256, 256))
model.eval()
res1 = model(template, search1)
model.train()
res2 = model(template, search1)
model.eval()
res3 = model(template, search1)
model.train()
res4 = model(template, search1)
model.eval()
res5 = model(template, search1)

res = (res1 == res2)

a = 1
