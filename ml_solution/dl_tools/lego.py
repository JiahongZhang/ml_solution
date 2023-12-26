import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import drop_path


class IndexEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()


class TwoLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features,
            hidden_features=None, 
            norm_layer=nn.LayerNorm, 
            norm_input=True, 
    ):
        super(TwoLP, self).__init__()
        hidden_features = in_features*2 if hidden_features is None else hidden_features
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


def extent_embeding(old_embed, ext_embed):
    old_vocab_length, old_embed_dim = old_embed.weight.shape
    is_embed_weight = hasattr(ext_embed, 'shape')
    ext_embed_num = ext_embed.shape[0] if is_embed_weight else ext_embed
    new_vocab_length = old_vocab_length+ext_embed_num
    new_embed = IndexEmbedding(new_vocab_length, old_embed_dim)

    # Setting device and type accordingly
    new_embed.to(
        old_embed.weight.device,
        dtype=old_embed.weight.dtype,
    )
    # Copying the old entries
    new_embed.weight.data[:old_vocab_length, :] = \
        old_embed.weight.data[:old_vocab_length, :]
    
    if is_embed_weight:
        new_embed.weight.data[old_vocab_length:, :] = ext_embed.weight.data

    return new_embed


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)



