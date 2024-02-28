import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_


class Embedding(nn.Module):
    def __init__(self, template_size, template_factor, embed_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(template_size, embed_dim, max_norm=1, norm_type=2.0)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.template_size = template_size
        self.template_factor = template_factor
        trunc_normal_(self.word_embeddings.weight, std=.02)

    def forward(self, template_anno):
        x, y, w, h = template_anno.unbind(1)
        template_w = self.template_size / self.template_factor * torch.sqrt(w / h)
        template_h = self.template_size / self.template_factor * torch.sqrt(h / w)
        wh = torch.stack((template_w, template_h), dim=1).int()
        wh = torch.clamp(wh, min=0, max=self.template_factor - 1)
        return self.word_embeddings(wh) + self.pos_embed


class SearchEmbedding(nn.Module):
    def __init__(self, search_size, embed_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(search_size + 1, embed_dim, max_norm=1, norm_type=2.0)
        self.search_size = search_size
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, embed_dim))
        trunc_normal_(self.word_embeddings.weight, std=.02)

    def forward(self, past_search_anno):
        past_search_anno = torch.clamp(past_search_anno, min=0, max=1)
        return self.word_embeddings((past_search_anno * self.search_size).int()) + self.pos_embed


if __name__ == '__main__':
    # embedding = Embedding(template_size=128, embed_dim=768, template_factor=2)
    # anno = torch.tensor(((-1, 1, 3, 4), (4, 2, 4, 3)))
    # print(embedding(anno))
    # print(embedding(anno).shape)
    embedding = SearchEmbedding(search_size=256, embed_dim=768)
    anno = torch.tensor(((-0.1, 0.2, 0.3, 1), (0.1, 0.2, 0.3, 0.4)))
    print(embedding(anno))
    print(embedding(anno).shape)
