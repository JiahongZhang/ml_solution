import urllib
from . import data_utils
import os 

config_pwd = os.path.dirname(os.path.abspath(__file__))


def _init():
    global config
    if os.path.isfile(f'{config_pwd}/config.json'):
        config = data_utils.json_load(f'{config_pwd}/config.json')
    else:
        config = {}

    if 'src_dir' not in config.keys():
        _src_dir = os.path.join(config_pwd, 'src')
        set('src_dir', _src_dir)

    os.makedirs(get('src_dir'), exist_ok=True)
    data_utils.json_write(config, f'{config_pwd}/config.json')


def set(key, value):
    try:
        config[key] = value
        data_utils.json_write(config, f'{config_pwd}/config.json')
        return True
    except KeyError:
        return False


def get(key):
    try:
        return config[key]
    except KeyError:
        return False


def reset():
    if os.path.isfile(f'{config_pwd}/config.json'):
        os.remove(f'{config_pwd}/config.json')
    _init()


def urllib_progress_bar(block_num, block_size, total_size):
    downloaded = block_num*block_size
    if downloaded < total_size:
        progress_num = f'{downloaded/1024**2:<.2f}/{total_size/1024**2:<.2f} MB'
        progress_percent = 100*downloaded/total_size
        progress_line = f"|{'#'*int(progress_percent//4):<25}|"
        print(f"{progress_line} {progress_num} - {progress_percent:5<.2f}%" , end='\r')
    else:
        print('\nDone!')


def download_from_modelscope(file_path, save_path=None, force=False):
    if save_path is None:
        save_path = f"{get('src_dir')}/{file_path}"
    if not os.path.isfile(save_path) or force:
        print(f'- Downloading {file_path}')
        scr_url = 'https://modelscope.cn/api/v1/models/hugo42/ml_solution/repo?Revision=master'
        full_url = f"{scr_url}&FilePath={file_path}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(full_url, save_path, urllib_progress_bar)

    


