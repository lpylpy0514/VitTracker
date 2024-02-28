import torch
from torch import nn
from timm.models.vision_transformer import Block, trunc_normal_
from lib.utils.pos_embed import get_sinusoid_encoding_table


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, img_size=224, patch_size=16):
        super().__init__()
        assert img_size % patch_size == 0
        self.proj = torch.nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=(patch_size, patch_size),
                                       stride=(patch_size, patch_size))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Preprocess(nn.Module):
    def __init__(self, template_size, patch_embedding, patch_size, num_heads=12,
                 mlp_ratio=4, depth=12, embed_dim=768, output_dim=4, sigmoid=True):
        super().__init__()
        self.patch_embed = patch_embedding
        num_patches = (template_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)for i in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.ln = nn.Linear(embed_dim, output_dim)
        if sigmoid is True:
            self.sig = nn.Sigmoid()
        else:
            self.sig = nn.Identity()
        self.output_dim = output_dim

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, z):
        B = z.shape[0]
        z = self.patch_embed(z)
        cls_token = self.cls_token.expand(B, -1, -1)
        feat = cls_token
        feat = torch.cat((feat, z), dim=1)
        feat = feat + self.pos_embed
        for i, blk in enumerate(self.blocks):
            feat = blk(feat)
        feat = self.norm(feat)
        feat = self.ln(feat[:, 0, :])
        color = feat[:, :self.output_dim - 1]
        transparency = self.sig(feat[:, self.output_dim - 1])
        return color, transparency


def build_preprocess(image_size, patch_size, embed_dim, depth, output_dim, sigmoid=True):
    patch_embedding = PatchEmbedding(embed_dim, image_size, patch_size)
    preprocess = Preprocess(image_size, patch_embedding, patch_size, embed_dim=embed_dim,
                            depth=depth, output_dim=output_dim, sigmoid=sigmoid)
    return preprocess


if __name__ == '__main__':
    template = torch.rand((1, 3, 128, 128))
    preprocess = build_preprocess(template_size=128, patch_size=16, embed_dim=768, depth=2, output_dim=4)
    color, transparency = preprocess(template)
    print(color)
    print(transparency)
