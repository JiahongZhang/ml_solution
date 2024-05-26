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


def plt_confusion_matrix(conf_matrix, title=None, x_labels='auto', y_labels='auto'):
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
        fmt = '',
        xticklabels=x_labels,
        yticklabels=y_labels
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


# text_attention_latex part BEGIN

def latex_sensitive_clean(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


def non_text_suppress(text_list, atten_weights):
	non_text = ',.!?:-，。？！：'
	replace_value = np.median(atten_weights) - np.std(atten_weights)
	for i, token in enumerate(text_list):
		for t in token:
			if t in non_text:
				atten_weights[i] = replace_value
				break
	return atten_weights
	

def text_attention_latex(
		text_list, # list of tokens string
		atten_weights, # list of attention weight
		latex_save_path, # .tex save path
		color='mycolor', 
		suppress=True # whether to suppress non text
		):
	assert(len(text_list) == len(atten_weights))
	word_num = len(text_list)
	text_list = latex_sensitive_clean(text_list)
	if suppress:
		 atten_weights = non_text_suppress(text_list, atten_weights)
	with open(latex_save_path,'w') as f:
		f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\definecolor{mycolor}{RGB}{0,127,255}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
		string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
		for idx in range(word_num):
			string += "\\colorbox{%s!%s}{"%(color, atten_weights[idx])+"\\strut " + text_list[idx]+"} "
		string += "\n}}}"
		f.write(string+'\n')
		f.write(r'''\end{CJK*}
\end{document}''')

# text_attention_latex part END

