# https://blog.csdn.net/taoqick/article/details/122029418
# Similarity of neural network representations revisited. -- Hinton

import math
import numpy as np


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

from lib.models.vittrack.vit import VisionTransformer, PatchEmbedding
import torch
from torch import nn
import cv2 as cv
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target

if __name__=='__main__':
    embed_dim = 256
    pemode = "conv2x2"
    patch_embed = PatchEmbedding(embed_dim=embed_dim, activation=nn.Hardswish(), img_size=256, patch_size=16, mode=pemode)
    model_mtr = VisionTransformer(template_size=128, search_size=256, patch_embedding=patch_embed, patch_size=16,
                              depth=12, embed_dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=4).eval()
    checkpoint_mtr = torch.load("/home/ymz/newdisk2/playground57/checkpoints/train/mtr/v_d12c256/MTR_ep0500.pth.tar")['net']
    del checkpoint_mtr['pos_embed']
    missing_keys, unexpected_keys = model_mtr.load_state_dict(checkpoint_mtr, strict=False)
    print("missing keys: ", missing_keys)

    model_finetune = VisionTransformer(template_size=128, search_size=256, patch_embedding=patch_embed, patch_size=16,
                              depth=12, embed_dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=4).eval()
    checkpoint_finetune = torch.load("/home/ymz/newdisk2/workspace_tracking/output/checkpoints/train/vittrack/d12c256conv2_mtr/VitTrack_ep0500.pth.tar")['net']
    for key in list(checkpoint_finetune.keys()):
        checkpoint_finetune[key[9:]] = checkpoint_finetune[key]
    missing_keys, unexpected_keys = model_finetune.load_state_dict(checkpoint_finetune, strict=False)
    print("missing keys: ", missing_keys)

    pre = Preprocessor()
    img = cv.imread("/home/ymz/newdisk1/ImageNet/train/n01440764/n01440764_18.JPEG")
    x1, y1, w, h = 98, 75, 112, 64
    # rect = cv.selectROI(img)
    # print(rect)
    # cv.destroyAllWindows()
    # x1, y1, w, h = rect
    x_patch_arr, resize_factor, x_amask_arr = sample_target(img, [x1, y1, w, h], 4,
                                                            output_sz=256)  # (x1, y1, w, h)
    search = pre.process(x_patch_arr, x_amask_arr).tensors.cpu()

    x_patch_arr, resize_factor, x_amask_arr = sample_target(img, [x1, y1, w, h], 2,
                                                            output_sz=128)  # (x1, y1, w, h)
    template = pre.process(x_patch_arr, x_amask_arr).tensors.cpu()


    # bs = 1
    # template = torch.randn((bs, 3, 128, 128))
    # search = torch.randn((bs, 3, 256, 256))
    feat_mtr = model_mtr(template, search)[-1].detach().numpy()[0]
    feat_finetune = model_finetune(template, search)[-1].detach().numpy()[0]

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(feat_mtr, feat_finetune)))
    # print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(feat_mtr, feat_finetune)))
    # print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))
