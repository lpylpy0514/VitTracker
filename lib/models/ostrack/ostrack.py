"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.ostrack.draw import Draw, Color, DrawMask, ExtraTemplateMask
from lib.models.ostrack.embedding import Embedding, SearchEmbedding
from lib.models.ostrack.preprocess import build_preprocess
from lib.models.ostrack.clipvit import clipvittracking_base_patch16


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", mode="teacher", channels=768,
                 template_preprocess=None, search_preprocess=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.template_preprocess = template_preprocess
        self.search_preprocess = search_preprocess
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                template_anno=None,
                past_search_anno=None,
                template_mask=None,
                ):
        # extra_tokens = None
        B, C, W, H = template.shape
        extra_features = {}
        if self.template_preprocess is not None:
            assert template_anno is not None, "need template annotations"
            if type(self.template_preprocess) is Draw:
                template = self.template_preprocess(template, template_anno)
            elif type(self.template_preprocess) is Embedding:
                extra_features['template_token'] = self.template_preprocess(template_anno)
            elif type(self.template_preprocess) is torch.nn.Conv2d:
                if template_mask is None:
                    # generate mesh-grid
                    assert H == W
                    image_size = H
                    indice = torch.arange(0, W).view(-1, 1)
                    coord_x = indice.repeat((image_size, 1)).view(image_size, image_size).float().to(template.device)
                    coord_y = indice.repeat((1, image_size)).view(image_size, image_size).float().to(template.device)
                    x1, y1, w, h = (template_anno.view(B, 4, 1, 1) * image_size).unbind(1)
                    x2, y2 = x1 + w, y1 + h
                    alpha_image = (x2 > coord_x) & (coord_x > x1) & (y2 > coord_y) & (coord_y > y1)
                    alpha_image = alpha_image.float().view(B, 1, H, W)
                else:
                    alpha_image = template_mask.view(B, 1, H, W)

                extra_features['template_alpha'] = self.template_preprocess(alpha_image).flatten(2).transpose(1, 2)
            elif type(self.template_preprocess) is DrawMask:
                template = self.template_preprocess(template, template_mask)
                # from lib.models.ostrack.draw import depreprocess
                # import cv2
                # image = depreprocess(template[0:1])
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            elif type(self.template_preprocess) is ExtraTemplateMask:
                extra_features['template_embedding'] = self.template_preprocess(template, template_mask)
        if self.search_preprocess is not None:
            assert past_search_anno is not None, "need search annotations"
            if type(self.search_preprocess) is Draw:
                search = self.search_preprocess(search, past_search_anno)
            elif type(self.search_preprocess) is Embedding:
                extra_features['template_embedding'] = self.search_preprocess(past_search_anno)

        # from lib.models.ostrack.draw import depreprocess
        # import cv2
        # image = depreprocess(template[0:1])
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x, aux_dict = self.backbone(z=template, x=search,
                                    extra_features=extra_features,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        # if self.mode == "student" and self.training:
        #     aux_dict = self.forward_aux(aux_dict)
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
        
    def forward_aux(self, aux_dict):
        for i, feature in enumerate(aux_dict['res_list']):
            feature = torch.transpose(feature, 1, 2)
            feature = self.convs[i](feature)
            feature = torch.transpose(feature, 1, 2)
            aux_dict['res_list'][i] = feature 
        return aux_dict


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'clipvittracking_base_patch16':
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        backbone = clipvittracking_base_patch16(pretrained, search_size=256, template_size=128)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    # if not pretrained.endswith('pt'):
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    # if cfg.MODEL.PREPROCESS == 'draw':
    #     color = Color(colormode="learn_color")
    #     preprocess = Draw(image_size=cfg.DATA.TEMPLATE.SIZE, drawmode='rect', color=color)
    # elif cfg.MODEL.PREPROCESS == 'draw_based_on_template':
    #     net = build_preprocess(image_size=cfg.DATA.TEMPLATE.SIZE, patch_size=16, embed_dim=768, depth=1, output_dim=4)
    #     color = Color(colormode="generate_color", net=net)
    #     preprocess = Draw(image_size=cfg.DATA.TEMPLATE.SIZE, drawmode='mask', color=color)
    # elif cfg.MODEL.PREPROCESS == 'draw_based_on_search':
    #     net = build_preprocess(image_size=cfg.DATA.SEARCH.SIZE, patch_size=16, embed_dim=768, depth=1, output_dim=4)
    #     color = Color(colormode="generate_color", net=net)
    #     preprocess = Draw(image_size=cfg.DATA.SEARCH.SIZE, drawmode='mask', color=color)
    # elif cfg.MODEL.PREPROCESS == 'template_embedding':
    #     preprocess = Embedding(template_size=cfg.DATA.TEMPLATE.SIZE, template_factor=cfg.DATA.TEMPLATE.FACTOR, embed_dim=768)
    # elif cfg.MODEL.PREPROCESS == 'search_embedding':
    #     preprocess = SearchEmbedding(search_size=cfg.DATA.SEARCH.SIZE, embed_dim=768)
    # else:
    #     preprocess = None
    if cfg.MODEL.PROCESS.TEMPLATE == "draw_based_on_template":
        net = build_preprocess(image_size=cfg.DATA.TEMPLATE.SIZE, patch_size=16, embed_dim=768, depth=1, output_dim=4)
        color = Color(colormode="generate_color", net=net)
        template_preprocess = Draw(image_size=cfg.DATA.TEMPLATE.SIZE, drawmode='mask', color=color)
    elif cfg.MODEL.PROCESS.TEMPLATE == "draw_based_on_template_nosig":
        net = build_preprocess(image_size=cfg.DATA.TEMPLATE.SIZE, patch_size=16, embed_dim=768, depth=1, output_dim=4, sigmoid=False)
        color = Color(colormode="generate_color", net=net)
        template_preprocess = Draw(image_size=cfg.DATA.TEMPLATE.SIZE, drawmode='mask', color=color)
    elif cfg.MODEL.PROCESS.TEMPLATE == "template_embedding":
        template_preprocess = Embedding(template_size=cfg.DATA.TEMPLATE.SIZE, template_factor=cfg.DATA.TEMPLATE.FACTOR,
                               embed_dim=768)
    elif cfg.MODEL.PROCESS.TEMPLATE == "template_alpha":
        template_preprocess = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        for param in template_preprocess.parameters():
            torch.nn.init.zeros_(param)
    elif cfg.MODEL.PROCESS.TEMPLATE == "template_draw_mask":
        net = build_preprocess(image_size=cfg.DATA.TEMPLATE.SIZE, patch_size=16, embed_dim=768, depth=1, output_dim=4)
        color = Color(colormode="generate_color", net=net)
        template_preprocess = DrawMask(image_size=cfg.DATA.TEMPLATE.SIZE, color=color)
    elif cfg.MODEL.PROCESS.TEMPLATE == "template_draw_mask_nosig":
        net = build_preprocess(image_size=cfg.DATA.TEMPLATE.SIZE, patch_size=16, embed_dim=768, depth=1, output_dim=4, sigmoid=False)
        color = Color(colormode="generate_color", net=net)
        template_preprocess = DrawMask(image_size=cfg.DATA.TEMPLATE.SIZE, color=color)
    elif cfg.MODEL.PROCESS.TEMPLATE == "extra_template_mask":
        patch_embedding = torch.nn.Conv2d(in_channels=3, out_channels=768, kernel_size=(16, 16), stride=(16, 16))
        template_preprocess = ExtraTemplateMask(patch_embedding, image_size=cfg.DATA.TEMPLATE.SIZE, patch_size=16, embed_dim=768)
    else:
        template_preprocess = None

    if cfg.MODEL.PROCESS.SEARCH == "draw_based_on_search":
        net = build_preprocess(image_size=cfg.DATA.SEARCH.SIZE, patch_size=16, embed_dim=768, depth=1, output_dim=4)
        color = Color(colormode="generate_color", net=net)
        search_preprocess = Draw(image_size=cfg.DATA.SEARCH.SIZE, drawmode='mask', color=color)
    elif cfg.MODEL.PROCESS.TEMPLATE == "search_embedding":
        search_preprocess = SearchEmbedding(search_size=cfg.DATA.SEARCH.SIZE, embed_dim=768)
    elif cfg.MODEL.PROCESS.TEMPLATE == "search_alpha":
        search_preprocess = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        for param in search_preprocess.parameters():
            torch.nn.init.zeros_(param)
    else:
        search_preprocess = None

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        template_preprocess=template_preprocess,
        search_preprocess=search_preprocess,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

def build_small_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           depth=3,
                                           channel=cfg.MODEL.BACKBONE.CHANNELS,
                                           heads=cfg.MODEL.BACKBONE.HEADS
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        mode='student',
        channels=cfg.MODEL.BACKBONE.CHANNELS
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
