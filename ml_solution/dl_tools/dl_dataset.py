from torch.utils.data import Dataset


def tokenizer_concat(main_tokenizer, sub_tokenizer):
    vocab_dict = sub_tokenizer.get_vocab()
    ext_token_num = main_tokenizer.add_tokens(list(vocab_dict.keys()))
    return main_tokenizer, ext_token_num


class JsonDataset(Dataset):
    def __init__(self, json, data_processor=None):
        self.data, self.indexs = [], []
        self.json = json
        for index in self.json.keys():
            self.indexs.append(index)
            self.data.append(self.json[index])
        
        self.data_processor = data_processor
        if hasattr(self.data_processor, 'preprocess'):
            self.data = [self.data_processor.preprocess(item) \
                         for item in self.data]

    
    def __getitem__(self, idx):
        item = self.data[idx]
        if hasattr(self.data_processor, 'postprocess'):
            item = self.data_processor.postprocess(item)
        return item

    
    def __len__(self):
        return len(self.data)
    


