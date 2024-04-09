import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
            self, 
            gamma=2, 
            reduction='mean', 
            weights=None
            ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights))
        else:
            self.weights = None
        self.reduction = reduction

    def forward(self, outputs, targets):
        if self.weights is not None:
            cross_entropy = F.cross_entropy(outputs, targets, self.weights, reduction='none')
        else:
            cross_entropy = F.cross_entropy(outputs, targets, reduction='none')
        probs = outputs.softmax(1)
        ground_probs = probs[range(len(targets)), targets]
        focal_loss = (1-ground_probs)**self.gamma * cross_entropy
        
        if self.reduction=='mean':
            return torch.mean(focal_loss)
        elif self.reduction=='sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        

class ScoreLoss(nn.Module):
    def __init__(
            self, k=1, b=0,
            reduction='mean'
            ):
        super(ScoreLoss, self).__init__()
        self.k, self.b = k, b
        self.reduction = reduction

    def forward(self, outputs, targets):
        targets = (targets - self.b) / self.k
        mse_loss = F.mse_loss(outputs.squeeze(), targets, reduction='none')
        
        if self.reduction=='none':
            return mse_loss
        elif self.reduction=='mean':
            return mse_loss.mean()
        elif self.reduction=='sum':
            return mse_loss.sum()


def softmax_CE(logits, labels):
    if len(labels.shape)==1:
        return F.cross_entropy(logits, labels)
    logits = logits.softmax(1)
    selected_logits = torch.masked_select(logits, labels==1)
    loss = -torch.log(selected_logits).sum() / torch.numel(selected_logits)
    return loss


class WeightScoreLoss(nn.Module):
    def __init__(self, 
                 k=1, b=0, reduction='mean', weight_name='weight', 
                 score_name='score', pred_score_name='score'):
        super(WeightScoreLoss, self).__init__()
        self.score_loss = ScoreLoss(k=k, b=b, reduction='none')
        self.reduction = reduction
        self.weight_name = weight_name
        self.score_name = score_name
        self.pred_score_name = pred_score_name

    def forward(self, outputs, targets):
        weight = targets[self.weight_name]
        score_loss = self.score_loss(
            outputs[self.pred_score_name], targets[self.score_name])
        weighted_mse_loss = weight * score_loss

        if self.reduction=='none':
            return weighted_mse_loss
        elif self.reduction=='mean':
            return weighted_mse_loss.mean()
        elif self.reduction=='sum':
            return weighted_mse_loss.sum()


class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.t = torch.nn.Parameter(torch.tensor([0.0]))
        self.CE_loss = softmax_CE

    def forward(self, vectors1, vectors2, v1_labels):
        # row vector matrix
        vectors1 = torch.nn.functional.normalize(vectors1)
        vectors2 = torch.nn.functional.normalize(vectors2)
        logits = torch.matmul(vectors1, vectors2.T) * torch.exp(self.t)
        loss1 = self.CE_loss(logits, v1_labels)
        loss2 = self.CE_loss(logits.T, v1_labels.T)

        return (loss1+loss2)/2


class DictInputWarpper(nn.Module):
    def __init__(self, module, input_name, target_name):
        super(DictInputWarpper, self).__init__()
        self.module = module
        self.input_name = input_name
        self.target_name = target_name

    def forward(self, dict_input, dict_targets):
        inputs = dict_input[self.input_name]
        targets = dict_targets[self.target_name]
        return  self.module(inputs, targets)


class DistillLoss(nn.Module):
    def __init__(self, distill_keys, task_loss=None, alpha=0.5, beta=0.5):
        super(DistillLoss, self).__init__()
        self.distill_keys = distill_keys
        self.task_loss = task_loss
        self.alpha, self.beta = alpha, beta
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        loss = 0
        for k in self.distill_keys:
            loss += self.mse(outputs[k], targets[k])
        if self.task_loss is not None:
            loss = self.alpha*loss + self.beta*self.task_loss(outputs, targets)
        return loss
    