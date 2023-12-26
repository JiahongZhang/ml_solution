from ml_solution import config
from transformers import BertTokenizer, BertModel

src_dir = config.get('src_dir')



def get_tokenizer(model_str, locally=False):
    if locally:
        tokenizer = BertTokenizer.from_pretrained(f'{src_dir}/models/{model_str}/tokenizer')
    else:
        tokenizer = BertTokenizer.from_pretrained(model_str)
    return tokenizer

