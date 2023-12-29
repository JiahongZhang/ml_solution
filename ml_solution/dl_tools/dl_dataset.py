import torch
from torch.utils.data import Dataset, sampler


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


class TensorDictCollateFunc():
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
            stacked_batch[key] = torch.stack(tensor_list) \
                if self.sp_keys is None or key not in self.sp_keys \
                else self.sp_keys_funcs[key](tensor_list)

        return stacked_batch


def torch_concat_batchs(batchs):
    if isinstance(batchs[0], (list, tuple)):
        transpose = zip(*batchs)
        return [torch_concat_batchs(samples) for samples in transpose]
    keys = batchs[0].keys()
    concated_batchs = {}
    for key in keys:
        tensor_list = [torch.as_tensor(item[key]) for item in batchs]
        concated_batchs[key] = torch.concat(tensor_list, axis=0)

    return concated_batchs


def dataset_label_count(dataset, label_name):
    label_list = []
    for _, target in dataset:
        label_list.append(target[label_name])
    labels = torch.tensor(label_list)
    return labels.unique(return_counts=True)


def dataset_samples_weight_by_label(dataset, label_name):
    labels, labels_count = dataset_label_count(dataset, label_name)
    labels, labels_count = labels.numpy(), labels_count.numpy()

    label_num_dict = {}
    for i, label in enumerate(labels):
        label_num_dict[label] = labels_count[i]

    weights = []
    for _, y in dataset:
        label = int(y[label_name])
        weights.append(1/(label_num_dict[label]*len(labels)))
    return torch.tensor(weights)


def dataset_label_balance_sampler(dataset, label_name, replacement=True):
    weights = dataset_samples_weight_by_label(dataset, label_name)
    balance_sampler = sampler.WeightedRandomSampler(weights, len(dataset), replacement=replacement)
    return balance_sampler

    
