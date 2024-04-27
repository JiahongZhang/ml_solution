import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ml_solution.dl_tools import engine, dl_dataset


def dl_predict(
        model, 
        dataloader, 
        grader=None, 
        device=engine.status['device']
        ):
    
    if grader is not None:
        grader.reset()
    model.to(device)
    model.eval()
    datas = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = engine.move_to_device(inputs, device)
            targets = engine.move_to_device(targets, device)
            outputs = model(inputs)
      
            if grader is not None:
                grader.update(targets=targets, outputs=outputs)

            outputs = engine.move_to_device(outputs, 'cpu')
            targets = engine.move_to_device(targets, 'cpu')
            datas.append((outputs, targets))
    datas = dl_dataset.torch_concat_batchs(datas)
    return datas


def plt_confusion_matrix(conf_matrix, title=None):
    percent = 100*conf_matrix / conf_matrix.sum(axis=1).reshape(-1,1)
    annot = []
    for i in range(conf_matrix.shape[0]):
        row = []
        for j in range(conf_matrix.shape[1]):
            row.append(f"{int(conf_matrix[i, j])}\n{percent[i, j]:.1f}%")
        annot.append(row)
    annot = np.array(annot)

    cmap = sns.cubehelix_palette(
        start=2.7, rot=0, dark=.2, light=.95, 
        hue=1, gamma=.8, reverse=False, as_cmap=True)
    sns.heatmap(
        conf_matrix, 
        annot=annot, 
        linewidth=.8, 
        cmap=cmap,
        annot_kws={"fontsize":'large'},
        fmt = ''
        )
    plt.ylabel('True Label')
    plt.xlabel('Predict Label')
    if title is not None:
        plt.title('Confusion Matrix - '+title)


def tensor_min_max_rescale(t):
    t = (t-t.min()) / (t.max()-t.min())
    return t


def multi_layer_atten_map(att_mat):
    '''
    Trainsform multilayer and multyhead [att_mat] to tokens to tokens [mul_atten]
    Input: 
        att_mat: Tensor [Layers, Head_nums, Token_nums, Token_nums]
    Output: 
        mul_atten: [Token_nums, Token_nums]
        mul_atten[i, j]: ith token's Q mul jth token's K 
    ref:
        https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map/notebook
        https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    '''

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1).detach()

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    #aug_att_mat = att_mat + residual_att
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    
    # Recursively multiply the weight matrices
    mul_atten = aug_att_mat[0]
    for i in range(1, aug_att_mat.shape[0]):
        mul_atten = torch.matmul(aug_att_mat[i], mul_atten)

    return mul_atten


def img_atten_map(pos_atten, img_size=(224, 224), sift_q=0.98):
    '''
    Reshape img [pos_atten] weight tensor to [img_size].
    Input: 
        pos_atten: [img_width*img_width] 1D tensor
    Output: 
        mul_atten: [Token_nums, Token_nums]
        mul_atten[i, j]: ith token's Q mul jth token's K 
    '''
    quantile = torch.quantile(pos_atten, sift_q)
    pos_atten = torch.clamp(pos_atten, min=0, max=quantile)
    width = int(np.sqrt(pos_atten.shape[0]))
    img_atten_map = pos_atten.reshape(-1, width, width)
    img_atten_map_ori = img_atten_map.squeeze()
    
    img_atten_map = img_atten_map.unsqueeze(0)
    img_atten_map = F.interpolate(img_atten_map, size=img_size, mode='bicubic')
    img_atten_map = img_atten_map.squeeze()

    return img_atten_map_ori, img_atten_map