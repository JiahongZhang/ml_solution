import random
import time
import os
from datetime import datetime
#import deepspeed
import torch
import torch.nn as nn
import numpy as np 
from ml_solution import data_utils


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


def model_state_dict(model):
    # not save DataParallel model
    if type(model) is torch.nn.DataParallel:
        return model.module.state_dict()
    else:
        return model.state_dict()


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


def change_to_half(obj):
    #move array/list/dict of tensor to device
    if torch.is_tensor(obj):
        if obj.type()=='torch.LongTensor':
            return obj
        else:
            return obj.half()
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = change_to_half(v)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(change_to_half(v))
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
            mix_pre=False
        ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

        self.device = device
        self.mix_pre = mix_pre
        self.scaler = torch.cuda.amp.GradScaler() if mix_pre else None


    def train(self):
        dataloader = self.dataloaders['train']
        self.model.train()
        for inputs, targets in dataloader:
            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)
            self.optimizer.zero_grad()
            
            if self.mix_pre:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
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


'''# deepspeed trainer developing
class DeepSpeedTrainer():
    def __init__(self, model, ds_args, dataloaders, criterion):
        deepspeed.init_distributed()
        # https://deepspeed.readthedocs.io/en/latest/initialize.html
        self.model_engine, self.optimizer, _, _ = \
                                deepspeed.initialize(
                                    args=ds_args,
                                    model=model,
                                    model_parameters=model.parameters()
                                    )
        self.dataloaders = dataloaders
        self.criterion = criterion


    def train(self):
        dataloader = self.dataloaders['train']
        self.model_engine.train()
        for inputs, targets in dataloader:
            inputs = change_to_half(inputs)
            inputs = move_to_device(inputs, self.model_engine.device)
            targets = move_to_device(targets, self.model_engine.device)
            #self.optimizer.zero_grad()
            
            outputs = self.model_engine(inputs)
            loss = self.criterion(outputs, targets)
            self.model_engine.backward(loss)
            self.model_engine.step()
            
            yield targets, outputs, loss


    def valid(self):
        dataloader = self.dataloaders['valid']
        self.model_engine.module.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = move_to_device(inputs, self.model_engine.device)
                targets = move_to_device(targets, self.model_engine.device)
                
                outputs = self.model_engine(inputs)
                loss = self.criterion(outputs, targets)
                yield targets, outputs, loss
'''

class TrainPipeline():
    def __init__(
            self, 
            trainer, 
            grader,
            logger=None,
            handler=None,
        ):
        self.trainer = trainer
        self.grader = grader
        self.logger = logger
        self.handler = handler
        

    def print_batch_progress(self, batch_count, phase, time_count):
        batch_num = len(self.trainer.dataloaders[phase])
        time_elapsed = time.time() - time_count
        process = 100 * batch_count / batch_num
        log = f'{phase:<5}: {batch_count:>5}/{batch_num:<5} \
            {time_elapsed:>3.2f}s/batch - {process:3.1f}%'
        
        print(log, end='\r')
        if process >= 100:
            print('')
        return log


    def print_epoch_progress(self, epoch_now, scores_dict):
        def dict_to_log(d):
            log_str = ''
            for k, v in d.items():
                log_str += f'{k:>5}: {v:<4.2f} '
            return log_str
        
        for phase in ['train', 'valid']:
            log = dict_to_log(scores_dict[phase])
            print(f'{phase} set: {log}')
        print(f'Epoch {epoch_now:<3}\n')
        return log


    def train_epoches(self, epoches):
        for i in range(1, epoches+1):
            scores_dict = {}
            for phase in ['train', 'valid']:
                batch_count = 1
                time_count = time.time()
                for targets, outputs, loss in eval(f'self.trainer.{phase}()'):
                    self.grader.update(targets=targets, outputs=outputs, loss=loss)
                    self.print_batch_progress(batch_count, phase, time_count)
                    time_count = time.time()
                    batch_count += 1

                scores = self.grader.compute()
                scores_dict[phase] = scores
                if self.logger is not None:
                    self.logger.log(i, phase, scores)
                self.grader.reset()

            self.print_epoch_progress(i, scores_dict)
            if self.handler is not None:
                self.handler.handle(i, scores_dict, self)
        
        if self.logger is not None and hasattr(self.logger, 'close'):
            self.logger.close()


class HandlerSaveModel():
    def __init__(self, metric_name, log_root, version, ideal_th=None):
        self.version = version
        self.log_root = log_root
        self.metric_name = metric_name
        self.best_scores = None
        self.ideal_scores = 'null'
        self.best_metric = 0
        self.ideal_th = ideal_th

        os.makedirs(f"{self.log_root}/{version}", exist_ok=True)
        

    def handle(
            self, epoch, 
            scores_dict, pipeline
        ):
        train_scores, valid_scores = scores_dict['train'], scores_dict['valid']
        train_metric, valid_metric = train_scores[self.metric_name], valid_scores[self.metric_name]

        if valid_metric>self.best_metric \
            or self.best_scores is None:
            self.best_metric = valid_metric
            self.best_scores = {
                'epoch': epoch,
                'train': train_scores,
                'valid': valid_scores,
            }
            model_state = model_state_dict(pipeline.trainer.model)
            torch.save(model_state, f'{self.log_root}/{self.version}/{self.version}_best.pt')

            if self.ideal_th is not None \
                and (train_metric-valid_metric)<=self.ideal_th:
                self.ideal_scores = self.best_scores
                torch.save(model_state, f'{self.log_root}/{self.version}/{self.version}_ideal.pt')
        
            records = {'best': self.best_scores, 'ideal':self.ideal_scores}
            data_utils.json_write(records, f'{self.log_root}/{self.version}/{self.version}_records.json')








