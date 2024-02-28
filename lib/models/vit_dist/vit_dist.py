import torch
from torch import nn
from timm.models.vision_transformer import Block
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
import importlib
import argparse


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1))


class LevitPatchEmbedding(nn.Module):
    def __init__(self, embed_dim, activation, img_size=224, patch_size=16):
        super().__init__()
        self.net = b16(embed_dim, activation)

    def forward(self, x):
        x = self.net(x).flatten(2).transpose(1, 2)
        return x


class OstrackDist(nn.Module):
    def __init__(self, patch_embedding, box_head, num_heads=4, mlp_ratio=4, depth=3, embed_dim=768, head_type="CENTER", mode="eval"):
        super().__init__()
        self.patch_embed = patch_embedding
        self.pos_embed_z = nn.Parameter(torch.zeros(1, 64, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.box_head = box_head
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.mode = mode
        if self.mode == 'training':
            self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=768, kernel_size=1) for i in range(depth)])
        elif self.mode == 'transformerblock':
            self.extblocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)for i in range(depth)])
            self.convs = nn.ModuleList([nn.Conv1d(in_channels=128, out_channels=768, kernel_size=1) for i in range(depth)])
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z, x):
        z = self.patch_embed(z)
        x = self.patch_embed(x)

        z += self.pos_embed_z
        x += self.pos_embed_x

        x = torch.cat((z, x), dim=1)

        distillation_list = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.mode != 'eval':
                dist_x = self.norm(x)
                distillation_list.append(dist_x)

        x = self.norm(x)

        out = self.forward_head(x, None)
        if self.mode != 'eval':
            out['res_list'] = self.forward_aux(distillation_list, self.mode)

        return out


    def forward_aux(self, distillation_list, mode):
        if mode == 'training':
            for i, feature in enumerate(distillation_list):
                feature = torch.transpose(feature, 1, 2)
                feature = self.convs[i](feature)
                feature = torch.transpose(feature, 1, 2)
                distillation_list[i] = feature
        elif mode == 'transformerblock':
            for i, feature in enumerate(distillation_list):
                feature = self.extblocks[i](feature)
                feature = torch.transpose(feature, 1, 2)
                feature = self.convs[i](feature)
                feature = torch.transpose(feature, 1, 2)
                distillation_list[i] = feature            
        else:
            pass
        return distillation_list


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
            # x = self.box_head(opt_feat, gt_score_map)
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

import math

def build_ostrack_dist(cfg, depth=3, mode='eval'):
    embed_dim = cfg.MODEL.BACKBONE.CHANNELS
    num_heads=cfg.MODEL.BACKBONE.HEADS
    patch_embedding = LevitPatchEmbedding(embed_dim, nn.Hardswish)
    box_head = build_box_head(cfg, embed_dim)
    model = OstrackDist(patch_embedding, box_head, num_heads=num_heads, mode=mode, embed_dim=embed_dim, depth=depth)
    if cfg.MODEL.PRETRAIN_FILE and mode != 'eval':
        if cfg.MODEL.PRETRAIN_FILE.endswith('pth'):
            ckpt = torch.load(cfg.MODEL.PRETRAIN_FILE)['model']#pth用model,tar用net
            pe = ckpt['pos_embed'][:, 1:, :]
            b_pe, hw_pe, c_pe = pe.shape
            side_pe = int(math.sqrt(hw_pe))
            pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0, 3, 1, 2])  # b,c,h,w
            side_num_patches_search = 16
            side_num_patches_template = 8
            pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search],
                                                align_corners=True, mode='bicubic')
            pe_s = torch.flatten(pe_s_2D.permute([0, 2, 3, 1]), 1, 2)
            pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template],
                                                align_corners=True, mode='bicubic')
            pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
            ckpt['pos_embed_z'] = pe_t
            ckpt['pos_embed_x'] = pe_s
            a, b = model.load_state_dict(ckpt, strict=False)
        elif cfg.MODEL.PRETRAIN_FILE.endswith('tar'):
            ckpt = torch.load(cfg.MODEL.PRETRAIN_FILE)['net']#pth用model,tar用net
            pe = ckpt['pos_embed'][:, 1:, :]
            pe_t = pe[:, 0:256, :]
            pe_s = pe[:, 256:, :]
            b_pe, hw_pe, c_pe = pe_t.shape
            side_pe = int(math.sqrt(hw_pe))
            pe_2D = pe_t.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0, 3, 1, 2])  # b,c,h,w
            side_num_patches_template = 8
            pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template],
                                                align_corners=True, mode='bicubic')
            pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
            ckpt['pos_embed_z'] = pe_t
            ckpt['pos_embed_x'] = pe_s
            a, b = model.load_state_dict(ckpt, strict=False)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, default='vit_dist', help='Name of the train script.')
    parser.add_argument('--config', type=str, default='experiments/ostrack/ostrack_distillation_123_128_h64.yaml', help="Name of the config file.")
    args = parser.parse_args()

    config_module = importlib.import_module("lib.config.%s.config" % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(args.config)
    model = build_ostrack_dist(cfg, mode='training')
    template = torch.randn((1, 3, 128, 128))
    search = torch.randn((1, 3, 256, 256))
    torch.onnx.export(model,  # model being run
                      (template, search),  # model input (a tuple for multiple inputs)
                      'vit_dist.onnx',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['template', 'search'],  # model's input names
                      output_names=['outputs_coord'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    res = model(template, search)
    # print(res)

