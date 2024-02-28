import argparse
import torch
import _init_paths
from lib.models.ostrack import build_small_ostrack
from lib.models.vit_dist import build_ostrack_dist
# from lib.config.ostrack.config import cfg, update_config_from_file
from lib.config.vit_dist.config import cfg, update_config_from_file
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn as nn
import torch.nn.functional as F
# for onnx conversion and inference
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
from lib.test.evaluation.environment import env_settings

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='stark_lightning_X_trt', help='script name')
    parser.add_argument('--config', type=str, default='baseline_rephead_4_lite_search5', help='yaml configure file name')
    args = parser.parse_args()
    return args

def get_data(bs=1, sz_x=256, sz_z=128):
    # img_x = torch.randn(bs, 3, sz_x, sz_x, requires_grad=True)
    # mask_x = torch.rand(bs, sz_x, sz_x, requires_grad=True) > 0.5
    # feat_vec_z = torch.randn(hw_z, bs, c, requires_grad=True)  # HWxBxC
    # mask_vec_z = torch.rand(bs, hw_z, requires_grad=True) > 0.5  # BxHW
    # pos_vec_z = torch.randn(hw_z, bs, c, requires_grad=True)  # HWxBxC
    img_x = torch.randn(bs, 3, sz_x, sz_x, requires_grad=False)
    img_z = torch.randn(bs, 3, sz_z, sz_z, requires_grad=False)
    return img_x, img_z

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        return score_map_ctr, size_map, offset_map

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

def build_head(in_channels, out_channels):
    return CenterPredictor(in_channels, out_channels, 16, 16)


from lib.models.vit_dist.vit_dist import LevitPatchEmbedding
from timm.models.vision_transformer import Mlp, LayerScale, DropPath, Final, use_fused_attn

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x.permute(1, 0, 2))
        # q, k, v = qkv.split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        q = qkv[:, :, :self.num_heads * self.head_dim].reshape(N, B * self.num_heads, self.head_dim).permute(1, 0, 2)
        k = qkv[:, :, self.num_heads * self.head_dim : 2 * self.num_heads * self.head_dim].reshape(N, B * self.num_heads, self.head_dim).permute(1, 0, 2)
        v = qkv[:, :, 2 * self.num_heads * self.head_dim:].reshape(N, B * self.num_heads, self.head_dim).permute(1, 0, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q / self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(0, 1).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
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
            self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=1024, kernel_size=1) for i in range(depth)])
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
            score_map_ctr, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            out = {'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_model(cfg):
    patch_embedding = LevitPatchEmbedding(cfg.MODEL.BACKBONE.CHANNELS, nn.Hardswish)
    box_head = build_head(cfg.MODEL.BACKBONE.CHANNELS, cfg.MODEL.HEAD.NUM_CHANNELS)
    model = OstrackDist(patch_embedding, box_head, mode='eval', embed_dim=cfg.MODEL.BACKBONE.CHANNELS, num_heads=cfg.MODEL.BACKBONE.HEADS)
    return model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    torch.set_num_threads(1)
    # options = onnxruntime.SessionOptions()
    # options.intra_op_num_threads = 1
    load_checkpoint = True
    save_name = "vttrack.onnx"
    # update cfg
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    # build the stark model

    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        if args.script == 'ostrack':
            model = build_small_ostrack(cfg, training=False)
            checkpoint_name = os.path.join(save_dir,
                                           "checkpoints/train/%s/%s/OSTrack_ep0300.pth.tar"
                                           % (args.script, args.config))
        else :
            model = build_model(cfg)
            checkpoint_name = os.path.join(save_dir,
                                           "checkpoints/train/%s/%s/OstrackDist_ep0160.pth.tar"
                                           % (args.script, args.config))
        a, b = model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=False)
    # transfer to test mode
    # model = repvgg_model_convert(model)
    model.eval()
    """ rebuild the inference-time model """
    # backbone = model.backbone
    # bottleneck = model.bottleneck
    # position_embed = model.pos_emb_x
    # transformer = model.transformer
    # box_head = model.box_head
    # box_head.coord_x = box_head.coord_x.cpu()
    # box_head.coord_y = box_head.coord_y.cpu()
    # torch_model = STARK(backbone, bottleneck, position_embed, transformer, box_head)
    torch_model = model
    print(torch_model)
    torch.save(torch_model.state_dict(), "complete.pth")
    # get the network input
    bs = 1
    sz_x = cfg.TEST.SEARCH_SIZE
    sz_z = cfg.DATA.TEMPLATE.SIZE
    print(bs, sz_x, sz_z)
    img_x, img_z = get_data(1, 256, 128)
    torch_outs = torch_model(img_z, img_x)

    from lib.models.vit_dist import build_ostrack_dist
    model_ori = build_ostrack_dist(cfg)
    a, b = model_ori.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=False)
    model_ori.eval()
    out_ori = model_ori(img_z, img_x)

    torch.onnx.export(torch_model,  # model being run
                      (img_z, img_x),  # model input (a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=14,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['template', 'search'],  # model's input names
                      output_names=['output1', 'output2', 'output3'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)
    print("creating session...")
    ort_session = onnxruntime.InferenceSession(save_name, providers=['CPUExecutionProvider'])
    print("execuation providers:")
    print(ort_session.get_providers())
    # compute ONNX Runtime output prediction
    """warmup (the first one running latency is quite large for the onnx model)"""

    img_x_cuda, img_z_cuda = img_x.cuda(), img_z.cuda()
    torch_outs = torch_model(img_z, img_x)
    # onnx inference
    ort_inputs = {'template': to_numpy(img_z),
                      'search': to_numpy(img_x),
                      }
    s_ort = time.time()
    ort_outs = ort_session.run(None, ort_inputs)


