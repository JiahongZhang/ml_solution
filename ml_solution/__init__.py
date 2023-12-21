import os
from . import(
    explorer,
    config,
    data_utils,
    dl_tools
)


pp_root = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(pp_root, 'src')
os.makedirs(_src_dir, exist_ok=True)

config._init()
config.set('src_dir', _src_dir)

