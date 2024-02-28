import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.ostrack import build_ostrack, build_small_ostrack
from lib.models.efficientvit import build_efficienttrack
from lib.models.vit_dist import build_ostrack_dist
# from lib.models.HiT import build_hit
from lib.models.vittrack import build_vittrack
from lib.models.mae.vit import mae_vit_l
# forward propagation related
from lib.train.actors import OSTrackActor, HiTActor
from lib.train.actors import OSTrackDistillationActor
from lib.train.actors import VTActor
from lib.train.actors import VtActor
CenterActor = VtActor
# for import modules
import importlib
from lib.models.vt.clipvit import clipvittracking_base_patch16

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

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

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir
    # cfg.depth = 3
    # Create network
    if settings.script_name == "ostrack":
        if "vipt" in cfg.MODEL.PROCESS.TEMPLATE:
            from lib.models.vipt.ostrack_prompt import build_viptrack
            net = build_viptrack(cfg)
        else:
            net = build_ostrack(cfg)
    elif settings.script_name == "HiT":
        net = build_hit(cfg)
    elif settings.script_name == 'vit_dist' and cfg.TRAIN.AUX_TYPE == "mean":
        net = build_ostrack_dist(cfg)
    elif settings.script_name == 'vit_dist' and cfg.TRAIN.AUX_TYPE == "Trblk":
        net = build_ostrack_dist(cfg, mode='transformerblock')
    elif settings.script_name == 'vit_dist' and cfg.TRAIN.TEACHER == "MAE-L":
        net = build_ostrack_dist(cfg, depth=12, mode='training')
    elif settings.script_name == 'vit_dist':
        net = build_ostrack_dist(cfg, mode='training')
    elif settings.script_name == "vt":
        net = build_ostrack_dist(cfg, depth=12, mode='eval')
    elif settings.script_name == "efficienttrack":
        net = build_efficienttrack(cfg, mode="train")
    elif settings.script_name == "vittrack":
        pass
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params in model:', n_parameters)
    # if settings.script_name == "vit_dist":
    #     if cfg.TRAIN.TEACHER == 'ostrack':
    #         if cfg.TRAIN.AUX_TYPE == 'None':
    #             pass
    #         else:
    #             cfg.MODEL.HEAD.NUM_CHANNELS = 256
    #             TeacherWeight = "OSTrack_ep0300.pth.tar"
    #             TeacherNet = build_ostrack(cfg, training=False)
    #             TeacherNet.load_state_dict(torch.load(TeacherWeight, map_location='cpu')['net'], strict=True)
    #             TeacherNet = TeacherNet.cuda()
    #             with torch.no_grad():
    #                 TeacherNet.eval()
    #     elif cfg.TRAIN.TEACHER == 'MAE-L':
    #         weight = "/home/lpy/OSTrack/pretrained_models/mae_pretrain_vit_large.pth"
    #         TeacherNet = mae_vit_l(weight)
    #         TeacherNet = TeacherNet.cuda()
    #         with torch.no_grad():
    #             TeacherNet.eval()
    #     elif cfg.TRAIN.TEACHER == 'clip':
    #         TeacherNet = clipvittracking_base_patch16(pretrained=True, search_size=256, template_size=128)
    #         TeacherNet.cuda()
    #         TeacherNet = TeacherNet.cuda()
    #         with torch.no_grad():
    #             TeacherNet.eval()
    #     n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #     print('number of params in model:', n_parameters)
        # n_parameters = sum(p.numel() for p in TeacherNet.parameters() if p.requires_grad)
        # print('number of params in teacher:', n_parameters)
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        # TeacherNet = DDP(TeacherNet, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    settings.model_preprocess = getattr(cfg.MODEL, "PREPROCESS", "None")
    # Loss functions and Actors
    if settings.script_name == "ostrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'aux': cfg.TRAIN.AUX_WEIGHT}
        actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "HiT":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = HiTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif cfg.TRAIN.AUX_TYPE == 'None':
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = VtActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "ostrack" or settings.script_name == "vit_dist":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'aux': cfg.TRAIN.AUX_WEIGHT}
        # actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        actor = OSTrackDistillationActor(teachernet=TeacherNet, net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "vt" or settings.script_name == "efficienttrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = VtActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "vittrack":
        if "CORNER" in cfg.MODEL.HEAD.TYPE:
            objective = {'giou': giou_loss, 'l1': l1_loss}
            loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
            actor = VTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
        elif "CENTER" in cfg.MODEL.HEAD.TYPE:
            focal_loss = FocalLoss()
            objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
            loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
            actor = CenterActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
