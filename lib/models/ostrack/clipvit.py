from collections import OrderedDict
import hashlib
import warnings
import urllib
from tqdm import tqdm
from typing import Union, List
import math
import torch
import torch.nn.functional as F
from torch import nn
import os

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, search_size: int, template_size: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.embed_dim = width
        self.output_dim = output_dim
        self.embed_dim_list = [output_dim]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches_search + self.num_patches_template + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward_features(self, images_list):

        class_embedding = self.class_embedding.unsqueeze(0).unsqueeze(0)
        B = images_list[0].shape[0]
        class_embedding = class_embedding.expand(B, -1, -1)
        xz = class_embedding + self.positional_embedding[0:1, :].to(images_list[0].dtype)
        for i in range(len(images_list)):
            x = images_list[i]
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            if i == 0:
                x = x + self.positional_embedding[1:self.num_patches_search + 1, :].to(x.dtype)
                xz = torch.cat([xz, x], dim=1)
            else:
                x = x + self.positional_embedding[self.num_patches_search + 1:, :].to(x.dtype)
                xz = torch.cat([xz, x], dim=1)
        xz = self.ln_pre(xz)

        # x = self.conv1(x)  # shape = [*, width, grid, grid]
        # z = self.conv1(z)
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # z = z.reshape(z.shape[0], z.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # z = z.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # xz = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x, z], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # xz = xz + self.positional_embedding.to(x.dtype)
        # xz = self.ln_pre(xz)

        xz = xz.permute(1, 0, 2)  # NLD -> LND
        xz = self.transformer(xz)  #if not set batch_first=True for attention, the first should be number, the second should be batchsize.
        xz = xz.permute(1, 0, 2)  # LND -> NLD     N is batch, L is number

        # x = self.ln_post(x[:, 0, :])
        xz = self.ln_post(xz)

        # if self.proj is not None:
        #     xz = xz @ self.proj
        return xz[:, 1:self.num_patches_search + 1, :] # NLD

    def forward(self, z, x, **kwargs):
        xz = self.forward_features([x, z])
        # x = self.head(x)
        out=[xz]
        aux_dict = []
        return out, aux_dict

def clipvittracking_base_patch32(pretrained=False, pretrain_type='default',
                                  search_size=364, template_size=182, **kwargs):
    patch_size = 32
    model  = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size,
        width=768,
        layers=12,
        heads=12,
        output_dim=512
    )
    if pretrained:
        load_pretrained(model, name="ViT-B/32")
    return model

def clipvittracking_base_patch16(pretrained=False, pretrain_type='default',
                                  search_size=364, template_size=182, **kwargs):
    patch_size = 16
    model  = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size,
        width=768,
        layers=12,
        heads=12,
        output_dim=512
    )
    if pretrained:
        load_pretrained(model, name="ViT-B/16")
    return model

def clipvittracking_large_patch14(pretrained=False, pretrain_type='default',
                                  search_size=364, template_size=182, **kwargs):
    patch_size = 14
    model  = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768
    )
    if pretrained:
        load_pretrained(model, name="ViT-L/14")
    return model

def clipvittracking_large_patch14_336px(pretrained=False, pretrain_type='default',
                                        search_size=364, template_size=182, **kwargs):
    patch_size = 14
    model  = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768
    )
    if pretrained:
        load_pretrained(model, name="ViT-L/14@336px")
    return model

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def load_pretrained(model, name,
                    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    jit: bool = False, download_root: str = None):

    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            state_dict = torch.jit.load(opened_file, map_location=device if jit else "cpu").state_dict()
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")


    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in state_dict:
    #         del state_dict[key]
    state_dict_load = OrderedDict()
    for key in state_dict.keys():
        if key[0:7] == 'visual.':
            state_dict_load[key[7:]] = state_dict[key]
    if 'proj' not in model.state_dict().keys():
        del state_dict_load['proj']

    # adjust position encoding
    cls_pe =  state_dict_load['positional_embedding'][0:1,:]
    pe = state_dict_load['positional_embedding'][1:,:]
    hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.num_patches_search))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))
    pe_2D = pe.reshape([side_pe, side_pe, c_pe]).permute([2,0,1]).unsqueeze(0)  #b,c,h,w
    if side_pe != side_num_patches_search:
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search], align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0,2,3,1]), 1, 2).squeeze(0)
    else:
        pe_s = pe
    if side_pe != side_num_patches_template:
        pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template], align_corners=True, mode='bicubic')
        pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2).squeeze(0)
    else:
        pe_t = pe
    pe_xz = torch.cat((cls_pe, pe_s, pe_t), dim=0)
    state_dict_load['positional_embedding'] = pe_xz

    # convert_weights(model)

    model.load_state_dict(state_dict_load)
