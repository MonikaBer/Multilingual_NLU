import numpy as np
from sklearn.metrics import f1_score
import torch

def f1_score_func(preds, labels):
    #print(preds)
    #print(labels)
    #exit()
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels, label_to_id):
    label_dict_inverse = {v: k for k, v in label_to_id.items()}

    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


def f1_score_func_QA(preds, labels):
    preds = np.swapaxes(preds, 0, 1)

    preds_flat = np.argmax(preds, axis = 2).flatten()
    labels_flat = labels.flatten()
    #print(np.shape(preds_flat))
    #print(np.shape(labels_flat))
    #exit()

    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class_QA(preds, labels, id_to_label):
    preds = np.swapaxes(preds, 0, 1)
    preds = torch.tensor(np.argmax(preds, axis = 2))
    labels = torch.tensor(labels)

    preds = torch.swapaxes(preds, 0, 1)
    labels = torch.swapaxes(labels, 0, 1)

    #print(preds.size())
    #print(labels.size())
    #exit()

    # iterate over indices
    equality_1 = []
    equality_2 = []
    for idx, (ind_pred, ind_lab) in enumerate(zip(preds, labels)):
        if(idx == 0):
            equality_1.append(torch.eq(ind_pred, ind_lab))
        elif(idx == 1):
            equality_1.append(torch.eq(ind_pred, ind_lab))
            equality_1 = torch.cat(equality_1)
            acc = torch.sum(equality_1)
            print(f'Entity 1 accuracy:\t{acc / len(equality_1)}')
        elif(idx == 2):
            equality_2.append(torch.eq(ind_pred, ind_lab))
        elif(idx == 3):
            equality_2.append(torch.eq(ind_pred, ind_lab))
            equality_2 = torch.cat(equality_2)
            acc = torch.sum(equality_2)
            print(f'Entity 2 accuracy:\t{acc / len(equality_2)}')

    #print(equality_1)
    #print(equality_2)
    full = torch.cat([equality_1, equality_2])
    acc = torch.sum(full)
    print(f'Both entities accuracy:\t{acc / len(full)}')
