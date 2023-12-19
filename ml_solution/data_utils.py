import pandas as pd
import json


def load_json(json_path):
    with open(json_path) as f:
        j = json.load(f)
    return j


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

