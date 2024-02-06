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
        self.weights = weights
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
        mse_loss = F.mse_loss(outputs.squeeze(), targets, reduction=self.reduction)
        return mse_loss


def softmax_CE(logits, labels):
    if len(labels.shape)==1:
        return F.cross_entropy(logits, labels)
    logits = logits.softmax(1)
    selected_logits = torch.masked_select(logits, labels==1)
    loss = -torch.log(selected_logits).sum() / torch.numel(selected_logits)
    return loss


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
