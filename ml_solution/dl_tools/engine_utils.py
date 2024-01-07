from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import wandb
from collections import OrderedDict


class ConfusionMetrics():
    def __init__(
            self, 
            num_classes=2, 
            metrics_list=["accuracy", "f1_score", "precision", "recall"], 
            eps=0.0001
            ):
        self.num_classes = num_classes
        self.compute_class = 1 if num_classes==2 else num_classes
        # conf_matrix[i, j]: true label is i, but predicted as j
        self.conf_matrix = np.zeros((num_classes, num_classes))
        self.sample_num = 0
        self.metrics_list = metrics_list
        self.eps = eps
        
    
    def accuracy(self):
        correct_num = 0
        for i in range(self.num_classes):
            correct_num += self.conf_matrix[i, i]
        acc = correct_num / self.sample_num
        return {'ACC': round(100*acc, 2)}


    def precision(self):
        precision_sum = 0
        for i in range(self.compute_class):
            precision_sum += self.conf_matrix[i, i] / (self.conf_matrix[:, i].sum()+self.eps)
        precision_sum /= self.compute_class
        return {'PRC': round(100*precision_sum, 2)}


    def recall(self):
        recall_sum = 0
        for i in range(self.compute_class):
            recall_sum += self.conf_matrix[i, i] / (self.conf_matrix[i, :].sum()+self.eps)
        recall_sum /= self.compute_class
        return {'REC': round(100*recall_sum, 2)}
    

    def f1_score(self):
        f1_sum = 0
        for i in range(self.compute_class):
            recall = self.conf_matrix[i, i] / (self.conf_matrix[i, :].sum()+self.eps)
            precision = self.conf_matrix[i, i] / (self.conf_matrix[:, i].sum()+self.eps)
            f1_sum += 2*precision*recall / (precision+recall+self.eps)
        f1_sum /= self.compute_class
        return {'F1': round(100*f1_sum, 2)}
    

    def update(self, x):
        label, label_pred = x['label'], x['label_pred']
        conf_matrix = confusion_matrix(label, label_pred, \
                                labels=range(self.num_classes))
        self.conf_matrix += conf_matrix
        self.sample_num += len(label)

    
    def compute(self):
        metrics_dict = {}
        for metric in self.metrics_list:
            metrics_dict.update(eval(f'self.{metric}()'))
        return metrics_dict


    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))
        self.sample_num = 0


class LossRecorder():
    def __init__(self, is_avg=True, loss_name='Loss'):
        self.loss_sum, self.sample_num = 0, 0
        self.loss_name = loss_name
        self.is_avg = is_avg
    
    def update(self, x):
        loss, sample_num = x['loss'], x['sample_num']
        total_loss = loss*sample_num if self.is_avg else loss
        self.loss_sum += total_loss
        self.sample_num += sample_num
    
    def compute(self):
        avg_loss = self.loss_sum / self.sample_num
        return {self.loss_name: round(avg_loss, 3)}

    def reset(self):
        self.loss_sum, self.sample_num = 0, 0


class Grader():
    def __init__(self, computers) -> None:
        self.computers = computers
    

    def update(self, **kwargs):
        targets = kwargs.get('targets')
        outputs = kwargs.get('outputs')
        loss = kwargs.get('loss')
        x = {
            'label': targets['label'].cpu().numpy(),
            'label_pred': outputs['logit'].argmax(1).cpu().detach().numpy(),
            'loss': loss.cpu().detach().numpy() if loss is not None else None
        }
        x['sample_num'] = len(x['label'])
        for computer in self.computers.values():
            computer.update(x)


    def compute(self):
        result = {}
        for computer in self.computers.values():
           result.update(computer.compute())
        return result
    
    
    def reset(self):
        for computer in self.computers.values():
            computer.reset()


class WandbLogger():
    def __init__(self, config, project):
        self.version = datetime.now().strftime(f'%y%m%d%H%M%S')
        wandb.init(config=config, name=self.version, project=project)
    
    def log(self, epoch, phase, scores):
        record = {}
        for k, v in scores.items():
            record[f'{phase}_{k}'] = v
        wandb.log(record, step=epoch-1)

    def close(self):
        wandb.finish()


class LocalLogger():
    def __init__(self, config):
        self.version = datetime.now().strftime(f'%y%m%d%H%M%S')
        
    
    def log(self, epoch, phase, scores):
        record = {}
        for k, v in scores.items():
            record[f'{phase}_{k}'] = v
        


def data_parallel_state_dict_recover(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    return new_state_dict


