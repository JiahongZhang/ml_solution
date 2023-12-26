import random
import torch
import torch.nn as nn
import numpy as np


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
            device='CPU',
        ):
        self.model = model
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


class TrainPipline():
    def __init__(
            self, 
            trainer, 
            grader,
            logger,
            ):
        self.trainer = trainer
        self.grader = grader
        self.logger = logger
        
    
    def train_epoches(self, epoches):
        for i in range(epoches):
            for targets, outputs, loss in self.trainer.train():
                self.grader.update(targets, outputs, loss)
            self.logger.log(i, 'train', self.grader.compute())
            self.grader.reset()

            for targets, outputs, loss in self.trainer.valid():
                self.grader.update(targets, outputs, loss)
            self.logger.log(i, 'valid', self.grader.compute())
            self.grader.reset()

