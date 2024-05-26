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


class ClipLoss(nn.Module):
    def __init__(self, t=0.0, fixed_t=False):
        super(ClipLoss, self).__init__()
        if not fixed_t:
            self.register_parameter('t', nn.Parameter(torch.Tensor([t])))
        else:
            self.register_buffer('t', torch.Tensor([t]))


    def contrast_llh_1d(self, logits, labels):
        same_label_marks = [torch.where(labels==labels[i], 1, 0) for i in range(len(labels))]
        rel_matrix = torch.stack(same_label_marks)
        likelyhood_loss, n = 0, len(logits)
        for i in range(n):
            pos_sample_logits = torch.masked_select(logits[i], rel_matrix[i]==1)
            neg_sample_logits = torch.masked_select(logits[i], rel_matrix[i]==0)

            pos_exp = pos_sample_logits.exp()
            neg_exp_sum = neg_sample_logits.exp().sum()
            likelyhood_loss += ((pos_exp+neg_exp_sum+0.001).log() - pos_sample_logits).sum()
        return likelyhood_loss/n


    def forward(self, vectors1, vectors2, v1_labels=None):
        # row vector matrix
        vectors1 = vectors1 / vectors1.norm(p=2, dim=-1, keepdim=True)
        vectors2 = vectors2 / vectors2.norm(p=2, dim=-1, keepdim=True)
        logits = torch.matmul(vectors1, vectors2.mT) * torch.exp(self.t)

        labels = torch.tensor(range(len(logits))).to(vectors1.device) if v1_labels is None else v1_labels
        loss1 = self.contrast_llh_1d(logits, labels)
        loss2 = self.contrast_llh_1d(logits.T, labels) if v1_labels is not None else 0
       
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
    


    