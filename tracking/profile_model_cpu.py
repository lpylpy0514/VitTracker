import os
import sys
import argparse
import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for test speed.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='ostrack', choices=['ostrack', 'vit_dist', 'efficienttrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='vitb_256_mae_ce_32x4_ep300', help='yaml configure file name')
    args = parser.parse_args()
    return args


def evaluate_vit(model, template, search):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search), custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_warmup = 500
    T_test = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_warmup):
            _ = model(template, search)
        start = time.time()
        for i in range(T_test):
            _ = model(template, search)
        end = time.time()
        avg_lat = (end - start) / T_test
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))


def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 500
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    # torch.set_num_threads(1)
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "ostrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_small_ostrack
        model = model_constructor(cfg, training=False)
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        evaluate_vit(model, template, search)
    elif args.script == 'vit_dist':
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_ostrack_dist
        model = model_constructor(cfg)
        model.eval()
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        evaluate_vit(model, template, search)
    elif args.script == 'efficienttrack':
        from lib.models.efficientvit.efficientvit import replace_batchnorm
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_efficienttrack
        model = model_constructor(cfg)
        model.eval()
        replace_batchnorm(model)
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        evaluate_vit(model, template, search)
    else:
        raise NotImplementedError
