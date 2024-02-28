import argparse
import torch
from lib.models.efficientvit import build_efficienttrack
from lib.config.efficienttrack.config import cfg, update_config_from_file
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
    parser.add_argument('--script', type=str, default='efficienttrack', help='script name')
    parser.add_argument('--config', type=str, default='GAF256', help='yaml configure file name')
    args = parser.parse_args()
    return args

def get_data(bs=1, sz_x=256, sz_z=128):
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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    torch.set_num_threads(1)
    # options = onnxruntime.SessionOptions()
    # options.intra_op_num_threads = 1
    load_checkpoint = True
    save_name = "output.onnx"
    # update cfg
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        if args.script == 'efficienttrack':
            model = build_efficienttrack(cfg)
            checkpoint_name = os.path.join(save_dir,
                                           "checkpoints/train/%s/%s/EfficientTrack_ep0300.pth.tar"
                                           % (args.script, args.config))
        a, b = model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=False)
    model.eval()
    print(model)
    torch.save(model.state_dict(), "complete.pth")
    # get the network input
    bs = 1
    sz_x = cfg.TEST.SEARCH_SIZE
    sz_z = cfg.DATA.TEMPLATE.SIZE
    img_x, img_z = get_data(1, 256, 128)
    torch_outs = model(img_z, img_x)

    torch.onnx.export(model,  # model being run
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
    """########## inference with the pytorch model ##########"""
    # forward the template
    N = 1
    # torch_model = torch_model.cuda()
    # torch_model.box_head.coord_x = torch_model.box_head.coord_x.cuda()
    # torch_model.box_head.coord_y = torch_model.box_head.coord_y.cuda()

    """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)
    print("creating session...")
    ort_session = onnxruntime.InferenceSession(save_name, providers=['CPUExecutionProvider'])
    print("execuation providers:")
    print(ort_session.get_providers())
    # compute ONNX Runtime output prediction
    """warmup (the first one running latency is quite large for the onnx model)"""
    # for i in range(50):
    #     # pytorch inference
    #     # img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda = \
    #     #     img_x.cuda(), mask_x.cuda(), feat_vec_z.cuda(), mask_vec_z.cuda(), pos_vec_z.cuda()
    #     img_x_cuda, img_z_cuda = img_x.cuda(), img_z.cuda()
    #     torch_outs = model(img_z, img_x)
    #     # onnx inference
    #     ort_inputs = {'template': to_numpy(img_z),
    #                   'search': to_numpy(img_x),
    #                   }
    #     s_ort = time.time()
    #     ort_outs = ort_session.run(None, ort_inputs)
    """begin the timing"""
    # t_pyt = 0  # pytorch time
    # t_ort = 0  # onnxruntime time
    #
    # print("pytorch model average latency", t_pyt/N*1000)
    # print("onnx model average latency:", t_ort/N*1000)
    # print(N / t_pyt, "FPS")
    # print(N / t_ort, "FPS")

    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_outs), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
