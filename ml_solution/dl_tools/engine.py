import random
import os
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np


status = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'gpu_count': torch.cuda.device_count(),
    'version': datetime.now().strftime(f'%y%m%d%H%M%S')
}


def fix_random_seed(random_seed=42):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def move_to_device(obj, device):
    #move array/list/dict of tensor to device
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")
    

class TorchTrainer():
    def __init__(
            self, 
            model, 
            dataloaders,
            criterion,
            optimizer,
            device=status['device'],
        ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def train(self):
        dataloader = self.dataloaders['train']
        self.model.train()
        for inputs, targets in dataloader:
            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)
            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()
            yield targets, outputs, loss


    def valid(self):
        dataloader = self.dataloaders['valid']
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                yield targets, outputs, loss


class TrainPipeline():
    def __init__(
            self, 
            trainer, 
            grader,
            logger,
            call_back=None,
        ):
        self.trainer = trainer
        self.grader = grader
        self.logger = logger
        self.call_back = call_back
        
    
    def train_epoches(self, epoches):
        for i in range(epoches):
            for targets, outputs, loss in self.trainer.train():
                self.grader.update(targets, outputs, loss)
            train_scores = self.grader.compute()
            self.logger.log(i, 'train', train_scores)
            self.grader.reset()

            for targets, outputs, loss in self.trainer.valid():
                self.grader.update(targets, outputs, loss)
            valid_scores = self.grader.compute()
            self.logger.log(i, 'valid', valid_scores)
            self.grader.reset()

            if self.call_back is not None:
                self.call_back(i, train_scores, valid_scores, self)
        
        self.logger.close()


class CallBackSaveModel():
    def __init__(self, metric_name, log_root, version) -> None:
        self.version = version
        self.log_root = log_root
        self.metric_name = metric_name
        self.best_scores = None
        self.best_metric = 0

        os.makedirs(f"{self.log_root}/{version}", exist_ok=True)
        

    def call_back_save_best(
            self, epoch, 
            train_scores, valid_scores, pipeline
        ):
        valid_metric = valid_scores[self.metric_name]
        if valid_metric>self.best_metric \
            or self.best_scores is None:
            self.best_score = valid_metric
            self.best_scores = {
                'epoch': epoch,
                'train': train_scores,
                'valid': valid_scores,
            }
            model_state = pipeline.trainer.model.state_dict()
            torch.save(model_state, \
                f'{self.log_root}/{self.version}/{self.version}_best.pt')







