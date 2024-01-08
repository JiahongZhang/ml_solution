# ML Solution

[pypi](https://pypi.org/project/ml-solution/) [github](https://github.com/JiahongZhang/ml_solution)

This package is used to quickly build a pipline for mechine learning.

## Quick Start

### Installation

Easy installation with pip:

```bash
pip intall ml_solution
```

### Code Sample

The code below shows how to use ml_solution to build flexible deep learning model frame quickly:

```python
import torch
from torch import optim
import torch.nn as nn
from dataset import creat_loader
import modeling
from ml_solution.dl_tools import engine, engine_utils, train_utils
from ml_solution import data_utils
from transformers import XLMRobertaTokenizer

train_config = data_utils.json_load('./train_config.json')
dataset_config = data_utils.json_load('./dataset_config.json')

train_loader = creat_loader(dataset_config['train_json_path'],)
valid_loader = creat_loader(dataset_config['valid_json_path'])
dataloaders = {
    'train':train_loader,
    'valid':valid_loader
}

model = modeling.get_model()
optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
criterion = train_utils.DictInputWarpper(nn.CrossEntropyLoss(), 'logit', 'label')

metric_grader = engine_utils.ConfusionMetrics(
    num_classes=4, 
    metrics_list=train_config['metrics_list']
    )
loss_grader = engine_utils.LossRecorder()
computers = {
    'conf_metrics': metric_grader, 
    'loss': loss_grader
}
grader = engine_utils.Grader(computers)


wandb_init_config = data_utils.json_manipulate_keys(
    train_config, 
    ['lr', 'batch_size', "architecture"], 
    keep=True
    )
wandb_init_config['criterion'] = criterion.module.__class__.__name__
wandb_init_config['optimizer'] = optimizer.__class__.__name__
logger = engine_utils.WandbLogger(
    config=wandb_init_config, project=train_config['project'])

handler = engine.HandlerSaveModel(
    metric_name="ACC", 
    log_root=train_config['log_root'], 
    version=logger.version,
    ideal_th=5
    )

trainer = engine.TorchTrainer(
    model, dataloaders, criterion, 
    optimizer, device=device, mix_pre=train_config['mix_pre']
    )

train_pipeline = engine.TrainPipeline(
    trainer, grader, logger, 
    handler=handler
    )

train_pipeline.train_epoches(train_config['epoches'])


```