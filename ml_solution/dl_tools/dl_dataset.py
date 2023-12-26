import torch
from torch.utils.data import Dataset


def tokenizer_concat(main_tokenizer, sub_tokenizer, token_sifter=None):
    vocab_dict = sub_tokenizer.get_vocab()
    tokens = list(vocab_dict.keys())
    if token_sifter is not None:
        tokens = [token for token in tokens if token_sifter(token)]
    ext_token_num = main_tokenizer.add_tokens(tokens)
    return main_tokenizer, ext_token_num


class JsonDataset(Dataset):
    def __init__(self, json, postprocessor=None):
        self.postprocessor = postprocessor
        self.data, self.indexs = [], []
        for index in json.keys():
            self.indexs.append(index)
            self.data.append(json[index])
        
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.postprocessor is not None:
            item = self.postprocessor.postprocess(item)
        return item

    
    def __len__(self):
        return len(self.data)


class TensorDictCollateFunc:
    def __init__(self, sp_keys=None, sp_keys_funcs=None):
        self.sp_keys = sp_keys
        self.sp_keys_funcs = sp_keys_funcs

    def collate_fn(self, batch):
        if isinstance(batch[0], (list, tuple)):
            transpose = zip(*batch)
            return [self.collate_fn(samples) for samples in transpose]
        keys = batch[0].keys()
        stacked_batch = {}
        for key in keys:
            tensor_list = [torch.as_tensor(item[key]) for item in batch]
            stacked_batch[key] = torch.stack(tensor_list) if key not in self.sp_keys \
                                    else self.sp_keys_funcs[key](tensor_list)

        return stacked_batch

