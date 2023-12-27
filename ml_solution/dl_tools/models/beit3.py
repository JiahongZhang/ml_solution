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
beit3_src_dir = f'{src_dir}/models/beit3'
beit3_config = {
    'beit3_src_dir': beit3_src_dir, 
    'tokenizer_path': f'{beit3_src_dir}/beit3.spm'
}


def get_tokenizer(tokenizer_path=beit3_config['tokenizer_path']):
    tokenizer = XLMRobertaTokenizer(tokenizer_path)
    return tokenizer


class Beit3Basic(nn.Module):
    def __init__(self, model_scale, pretrain=False):
        super(Beit3Basic, self).__init__()
        self.config = eval(f'beit3_utils._get_{model_scale}_config()')
        self.beit3 = BEiT3(self.config)
        embed_dim = self.config.encoder_embed_dim
        self.img_embed_size = 8192
        self.mlm_head = nn.Linear(embed_dim, self.config.vocab_size)
        self.mim_head = nn.Linear(embed_dim, self.img_embed_size)
        if pretrain:
            pretrain_path = f'{beit3_src_dir}/beit3_{model_scale}_patch16_224.pth'
            self.load_state_dict(torch.load(pretrain_path)['model'])
        else:
            self.apply(lego.init_weights)


    









