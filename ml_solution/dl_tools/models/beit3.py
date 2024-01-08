import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer
from torchscale.model.BEiT3 import BEiT3
from ml_solution import config, dl_tools
from ml_solution.dl_tools import lego
from . import beit3_utils

model_card={
    'vocab_size': 64002,
}

src_dir = config.get('src_dir')


def get_tokenizer(tokenizer_path='models/beit3/beit3.spm', force_download=False):
    config.download_from_modelscope(tokenizer_path, force=force_download)
    tokenizer = XLMRobertaTokenizer(f"{src_dir}/{tokenizer_path}")
    return tokenizer


class Beit3Basic(nn.Module):
    def __init__(self, model_scale, pretrain=False, force_download=False):
        super(Beit3Basic, self).__init__()
        self.config = eval(f'beit3_utils._get_{model_scale}_config()')
        self.beit3 = BEiT3(self.config)
        embed_dim = self.config.encoder_embed_dim
        self.img_embed_size = 8192
        self.mlm_head = nn.Linear(embed_dim, self.config.vocab_size)
        self.mim_head = nn.Linear(embed_dim, self.img_embed_size)
        if pretrain:
            pretrain_path = f'models/beit3/beit3_{model_scale}_patch16_224.pth'
            config.download_from_modelscope(pretrain_path, force=force_download)
            self.load_state_dict(torch.load(f"{src_dir}/{pretrain_path}")['model'])
        else:
            self.apply(lego.init_weights)


    









