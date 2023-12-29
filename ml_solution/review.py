import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
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
    percent = 100*conf_matrix / conf_matrix.sum(axis=1)
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

        