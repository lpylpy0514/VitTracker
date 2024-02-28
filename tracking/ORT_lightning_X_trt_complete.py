import argparse
import torch
import _init_paths
# from lib.models.stark.repvgg import repvgg_model_convert
# from lib.models.stark import build_stark_lightning_x_trt
from lib.models.ostrack import build_small_ostrack
from lib.models.vit_dist import build_ostrack_dist
from lib.config.ostrack.config import cfg, update_config_from_file
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
    img_x = torch.randn(bs, 3, sz_x, sz_x, requires_grad=True)
    img_z = torch.randn(bs, 3, sz_z, sz_z, requires_grad=True)
    return img_x, img_z


# class STARK(nn.Module):
#     def __init__(self, backbone, bottleneck, position_embed, transformer, box_head):
#         super(STARK, self).__init__()
#         self.backbone = backbone
#         self.bottleneck = bottleneck
#         self.position_embed = position_embed
#         self.transformer = transformer
#         self.box_head = box_head
#         self.feat_sz_s = int(box_head.feat_sz)
#         self.feat_len_s = int(box_head.feat_sz ** 2)
#
#     def forward(self, img: torch.Tensor, mask: torch.Tensor,
#                 feat_vec_z: torch.Tensor, mask_vec_z: torch.Tensor, pos_vec_z: torch.Tensor):
#         # run the backbone
#         feat = self.bottleneck(self.backbone(img))  # BxCxHxW
#         mask_down = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
#         pos_embed = self.position_embed(bs=1)  # 1 is the batch-size. output size is BxCxHxW
#         # adjust shape
#         feat_vec_x = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
#         pos_vec_x = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
#         mask_vec_x = mask_down.flatten(1)  # BxHW
#         # concat with the template-related results
#         feat_vec = torch.cat([feat_vec_z, feat_vec_x], dim=0)
#         mask_vec = torch.cat([mask_vec_z, mask_vec_x], dim=1)
#         pos_vec = torch.cat([pos_vec_z, pos_vec_x], dim=0)
#         # get q, k, v
#         q = feat_vec_x + pos_vec_x
#         k = feat_vec + pos_vec
#         v = feat_vec
#         key_padding_mask = mask_vec
#         # run the transformer encoder
#         memory = self.transformer(q, k, v, key_padding_mask=key_padding_mask)
#         fx = memory[-self.feat_len_s:].permute(1, 2, 0).contiguous()  # (B, C, H_x*W_x)
#         fx_t = fx.view(*fx.shape[:2], self.feat_sz_s, self.feat_sz_s).contiguous()  # fx tensor 4D (B, C, H_x, W_x)
#         # run the corner head
#         outputs_coord = box_xyxy_to_cxcywh(self.box_head(fx_t))
#         return outputs_coord
#
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

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

    # def forward(self, x, gt_score_map=None):
    #     """ Forward pass with input x. """
    #     score_map_ctr, size_map, offset_map = self.get_score_map(x)
    #
    #     # assert gt_score_map is None
    #     # if gt_score_map is None:
    #     #     bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
    #     # else:
    #     #     bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)
    #
    #     return [score_map_ctr, torch.cat((size_map, offset_map), dim=1)]
    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # # assert gt_score_map is None
        # if gt_score_map is None:
        #     bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        # else:
        #     bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, size_map, offset_map


    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = torch.zeros((idx.shape[0], 2, 1), dtype=torch.int64) + idx.unsqueeze(1)
        size = size_map.flatten(2)[..., idx[0, 0]]
        offset = offset_map.flatten(2)[..., idx[0, 0]].squeeze(-1)

        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

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


from lib.models.vit_dist.vit_dist import LevitPatchEmbedding, OstrackDist



def build_model(cfg):
    patch_embedding = LevitPatchEmbedding(cfg.MODEL.BACKBONE.CHANNELS, nn.Hardswish)
    box_head = build_head(cfg.MODEL.BACKBONE.CHANNELS, cfg.MODEL.HEAD.NUM_CHANNELS)
    model = OstrackDist(patch_embedding, box_head, mode='eval', embed_dim=cfg.MODEL.BACKBONE.CHANNELS)
    return model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    torch.set_num_threads(1)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    load_checkpoint = True
    save_name = "complete.onnx"
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
                                           "checkpoints/train/%s/%s/OstrackDist_ep0300.pth.tar"
                                           % (args.script, args.config))
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=False)
    # transfer to test mode
    # model = repvgg_model_convert(model)
    model.eval()
    """ rebuild the inference-time model """
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
    torch.onnx.export(torch_model,  # model being run
                      (img_z, img_x),  # model input (a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['template', 'search'],  # model's input names
                      output_names=['output1', 'output2'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    """########## inference with the pytorch model ##########"""
    # forward the template
    N = 1000
    # torch_model = torch_model.cuda()
    # torch_model.box_head.coord_x = torch_model.box_head.coord_x.cuda()
    # torch_model.box_head.coord_y = torch_model.box_head.coord_y.cuda()

    """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)
    print("creating session...")
    ort_session = onnxruntime.InferenceSession(save_name, providers=['CPUExecutionProvider'])
    # ort_session.set_providers(["TensorrtExecutionProvider"],
    #                   [{'device_id': '1', 'trt_max_workspace_size': '2147483648', 'trt_fp16_enable': 'True'}])
    print("execuation providers:")
    print(ort_session.get_providers())
    # compute ONNX Runtime output prediction
    """warmup (the first one running latency is quite large for the onnx model)"""
    for i in range(50):
        # pytorch inference
        # img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda = \
        #     img_x.cuda(), mask_x.cuda(), feat_vec_z.cuda(), mask_vec_z.cuda(), pos_vec_z.cuda()
        img_x_cuda, img_z_cuda = img_x.cuda(), img_z.cuda()
        torch_outs = torch_model(img_z, img_x)
        # onnx inference
        ort_inputs = {'template': to_numpy(img_z),
                      'search': to_numpy(img_x),
                      }
        s_ort = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
    """begin the timing"""
    t_pyt = 0  # pytorch time
    t_ort = 0  # onnxruntime time

    # for i in range(N):
    #     # generate data
    #     img_x, img_z = get_data(bs=bs, sz_x=sz_x, sz_z=sz_z)
    #     # pytorch inference
    #     # img_x_cuda, img_z_cuda = img_x.cuda(), img_z.cuda()
    #     s_pyt = time.time()
    #     torch_outs = torch_model(img_z, img_x)
    #     e_pyt = time.time()
    #     lat_pyt = e_pyt - s_pyt
    #     t_pyt += lat_pyt
    #     # print("pytorch latency: %.2fms" % (lat_pyt * 1000))
    #     # onnx inference
    #     ort_inputs = {'template': to_numpy(img_z),
    #                   'search': to_numpy(img_x),
    #                   }
    #     s_ort = time.time()
    #     ort_outs = ort_session.run(None, ort_inputs)
    #     e_ort = time.time()
    #     lat_ort = e_ort - s_ort
    #     t_ort += lat_ort

    for i in range(N):
        img_x, img_z = get_data(bs=bs, sz_x=sz_x, sz_z=sz_z)
        # pytorch inference
        # img_x_cuda, img_z_cuda = img_x.cuda(), img_z.cuda()
        s_pyt = time.time()
        torch_outs = torch_model(img_z, img_x)
        e_pyt = time.time()
        lat_pyt = e_pyt - s_pyt
        t_pyt += lat_pyt
    for i in range(N):
        ort_inputs = {'template': to_numpy(img_z),
                          'search': to_numpy(img_x),
                          }
        s_ort = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        e_ort = time.time()
        lat_ort = e_ort - s_ort
        t_ort += lat_ort
        # print("onnxruntime latency: %.2fms" % (lat_ort * 1000))
    print("pytorch model average latency", t_pyt/N*1000)
    print("onnx model average latency:", t_ort/N*1000)
    print(N / t_pyt, "FPS")
    print(N / t_ort, "FPS")

    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_outs), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
