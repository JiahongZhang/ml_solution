import torch
import torch.nn as nn
import torch.nn.functional as F



class IndexEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()



def extent_embeding(old_embed, ext_embed=None):
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


