import torch
from torch import nn
from torch.nn import Module

from models.stylegan2.model import EqualLinear, PixelNorm
from transformers import CLIPTokenizer, CLIPTextModel

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]


class Mapper(Module):

    def __init__(self, opts, latent_dim=512):
        super(Mapper, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x


class SingleMapper(Module):

    def __init__(self, opts):
        super(SingleMapper, self).__init__()

        self.opts = opts

        self.mapping = Mapper(opts)

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(Module):

    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(opts)

    def forward(self, x):
        # print(x.shape)
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        # print(out.shape)
        # exit()
        return out

class FullStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(FullStyleSpaceMapper, self).__init__()

        self.opts = opts

        for c, c_dim in enumerate(STYLESPACE_DIMENSIONS):
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=c_dim))

    def forward(self, x):
        out = []
        for c, x_c in enumerate(x):
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            out.append(x_c_res)

        return out


class WithoutToRGBStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(WithoutToRGBStyleSpaceMapper, self).__init__()

        self.opts = opts

        indices_without_torgb = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
        self.STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in indices_without_torgb]

        for c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=STYLESPACE_DIMENSIONS[c]))

    def forward(self, x):
        out = []
        for c in range(len(STYLESPACE_DIMENSIONS)):
            x_c = x[c]
            if c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            else:
                x_c_res = torch.zeros_like(x_c)
            out.append(x_c_res)

        return out

class AttentionMapper(Module):
    def __init__(self, opts):
        super(AttentionMapper, self).__init__()
        self.opts = opts
        self.ca = CrossAttention()
        self.norm = PixelNorm()
        self.text_embedder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self,w,t):
        w = self.norm(w)
        t = self.text_embedder(t).last_hidden_state
        w = w.permute(1,0,2)
        t = t.permute(1,0,2)
        w = self.ca(w,t)
        w = self.ca(w,t)
        w = w.permute(1,0,2)
        t = t.permute(1,0,2)
        return w
        

class CrossAttention(Module):
    def __init__(self, dim=512, heads=4):
        super(CrossAttention, self).__init__()
        self.dim = dim        
        self.attention = nn.MultiheadAttention(dim, heads)
        self.ln = nn.LayerNorm([dim])
        self.ff = nn.Sequential(
            nn.LayerNorm([dim]),
            nn.Linear(dim,dim),
            nn.GELU(),
            nn.Linear(dim,dim),
        )

    def forward(self, w, t):
        # size = x.shape[-1]
        # x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        
        attention_value, _ = self.attention(w, t, t)
        attention_value = attention_value + w
        attention_value = self.ln(attention_value)
        attention_value = self.ff(attention_value) + attention_value
        attention_value = self.ln(attention_value)
        return attention_value
        # return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)