import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig

from ml_solution import config
from . import beit3_utils

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
        super().__init__()
        self.config = eval(f'beit3_utils._get_{model_scale}_config()')
        self.beit3 = BEiT3(self.config)
        embed_dim = self.config.encoder_embed_dim
        self.img_embed_size = 8192
        self.mlm_head = nn.Linear(embed_dim, self.config.vocab_size)
        self.mim_head = nn.Linear(embed_dim, self.img_embed_size)
        if pretrain:
            pretrain_path = f'{beit3_src_dir}/beit3_{model_scale}_patch16_224.pth'
            self.load_state_dict(torch.load(pretrain_path)['model'])




class Beit3Classifier(nn.Module):
    def __init__(self, classes_num=2, pretrain=False, classifier_embed=False):
        super().__init__()
        self.classifier_embed = classifier_embed
        beit3_base = Beit3Basic(pretrain)
        self.beit3 = beit3_base.beit3
        embed_dim = 768
        classifier_hidden_dim = 2048
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.Linear(classifier_hidden_dim, classes_num)
            )


    def forward(self, x):
        outputs = self.beit3(**x)
        x = outputs['encoder_out'][:, [0, 197], :].mean(axis=1)
        classifier_embed = None
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if self.classifier_embed and i==0:
                classifier_embed = x
        
        return {
            'logits': x,
            'token_embeds': outputs['encoder_out'],
            'classifier_embed': classifier_embed
            }







