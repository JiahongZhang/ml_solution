import json
import copy
from PIL import Image
import pandas as pd


def json_load(json_path):
    with open(json_path) as f:
        j = json.load(f)
    return j


def json_write(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f)


def json_manipulate_keys(item, ref_keys, keep=False):
    item = copy.deepcopy(item)
    keys = list(item.keys())
    for key in keys:
        exclude_flag = key not in ref_keys if keep else key in ref_keys
        if exclude_flag:
            del item[key]
    return item


def df_sample(
        df, sample_sig, 
        balance_on=None, 
        drop_index=True
    ):
    sample_num = int(len(df)*sample_sig) if sample_sig<1 else sample_sig
    if balance_on is None:
        sample1 = df.sample(sample_num)
        sample2 = df[~df.index.isin(sample1.index)]
    else:
        keys = df[balance_on].unique()
        avg_sample_num = int(sample_num/len(keys))
        remain_num = sample_num % len(keys)
        key_sampled_df_list = []
        for i, key in enumerate(keys):
            key_sample_num = avg_sample_num+1 if i<remain_num else avg_sample_num
            key_sampled_df = df[df[balance_on]==key].sample(key_sample_num)
            key_sampled_df_list.append(key_sampled_df)
        key_sampled_df = pd.concat(key_sampled_df_list)
        sample2 = df[~df.index.isin(key_sampled_df.index)]
        sample1 = key_sampled_df

    return sample1.reset_index(drop=drop_index), sample2.reset_index(drop=drop_index)


def img_pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def img_permute(img, redo=False):
    img = img.transpose(1, 2, 0) if redo else img.transpose(2, 0, 1)
    return img


def str_is_all_chinese(s):
    for _char in s:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


