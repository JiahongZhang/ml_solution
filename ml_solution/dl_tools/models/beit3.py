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

beit3_file_paths = {
    'tokenizer_vanilla': 'models/beit3/beit3.spm',
    'tokenizer_enzh_download': [
        'models/beit3/with_hfl_chinese-bert-wwm-ext_tokenizer/added_tokens.json',
        'models/beit3/with_hfl_chinese-bert-wwm-ext_tokenizer/sentencepiece.bpe.model',
        'models/beit3/with_hfl_chinese-bert-wwm-ext_tokenizer/special_tokens_map.json',
        'models/beit3/with_hfl_chinese-bert-wwm-ext_tokenizer/tokenizer_config.json',
        ],
    'tokenizer_enzh':'models/beit3/with_hfl_chinese-bert-wwm-ext_tokenizer',
    'model_base': 'models/beit3/beit3_base_patch16_224.pth',
    'model_large': 'models/beit3/beit3_large_patch16_224.pth'
}


src_dir = config.get('src_dir')


def get_tokenizer( 
        force_download=False,
        language=None
        ):
    if language is None:
        config.download_from_modelscope(beit3_file_paths['tokenizer_vanilla'], force=force_download)
        tokenizer_path = beit3_file_paths['tokenizer_vanilla']
        tokenizer = XLMRobertaTokenizer(f"{src_dir}/{tokenizer_path}")
    elif language == 'enzh':
        for file_path in beit3_file_paths['tokenizer_enzh_download']:
            config.download_from_modelscope(file_path, force=force_download)
        tokenizer_path = beit3_file_paths['tokenizer_enzh']
        tokenizer = XLMRobertaTokenizer.from_pretrained(f"{src_dir}/{tokenizer_path}")

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
            pretrain_path = beit3_file_paths[f'model_{model_scale}']
            config.download_from_modelscope(pretrain_path, force=force_download)
            self.load_state_dict(torch.load(f"{src_dir}/{pretrain_path}")['model'])
        else:
            self.apply(lego.init_weights)












